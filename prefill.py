"""
Curator — Prefill from Conversation History
Seeds the conception space from Claude Code history or a Claude.ai export.

Usage:
  # Claude Code local history
  python3 prefill.py                         # all projects
  python3 prefill.py --project myapp        # specific project
  python3 prefill.py --limit 50             # last N conversation files

  # Claude.ai export (conversations.json from Settings → Privacy → Export)
  python3 prefill.py --claudeai conversations.json
  python3 prefill.py --claudeai conversations.json --limit 20

  # Preview without storing (works without API key)
  python3 prefill.py --dry-run
  python3 prefill.py --claudeai conversations.json --dry-run

The script only processes your messages — not Claude's responses.
You are what you said, not what was said back to you.
"""

import os
import sys
import json
import glob
import argparse
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "curator"))
from schema import connect, surface, SignalQuality
from observe import observe, evaluate_signal_quality, batch_observe

# ─── Colors ──────────────────────────────────────────────────────────────────

GREEN  = "\033[92m"
BLUE   = "\033[94m"
YELLOW = "\033[93m"
DIM    = "\033[2m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

CLAUDE_DIR = os.path.expanduser("~/.claude/projects")
DB_PATH    = os.path.join(os.path.dirname(__file__), "curator_mcp.db")

# ─── JSONL reader ─────────────────────────────────────────────────────────────

def read_jsonl(path: str) -> list[dict]:
    messages = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    messages.append(obj)
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass
    return messages


def read_claudeai_export(path: str, limit: int = None) -> list[tuple[str, list[str]]]:
    """
    Read Claude.ai conversations.json export.
    Returns list of (conversation_name, [user_message_texts]).
    Sorted newest first.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Sort by updated_at, newest first
    data.sort(key=lambda c: c.get("updated_at", ""), reverse=True)

    if limit:
        data = data[:limit]

    result = []
    for convo in data:
        name = convo.get("name", "unknown")
        messages = convo.get("chat_messages", [])
        texts = []
        for msg in messages:
            if msg.get("sender") != "human":
                continue
            text = msg.get("text", "").strip()
            if text:
                texts.append(text)
        if texts:
            result.append((name, texts))

    return result


def extract_user_messages(messages: list[dict]) -> list[str]:
    """
    Pull user text from JSONL entries.
    Claude Code JSONL has varying structures — handle the common ones.
    """
    texts = []
    for msg in messages:
        role = msg.get("role", "")
        if role != "human" and role != "user":
            continue

        content = msg.get("content", "")

        # Content can be string or list of blocks
        if isinstance(content, str):
            text = content.strip()
            if text:
                texts.append(text)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "").strip()
                    if text:
                        texts.append(text)

    return texts


def find_jsonl_files(project: str = None, limit: int = None) -> list[str]:
    """Find JSONL files in ~/.claude/projects/"""
    if not os.path.exists(CLAUDE_DIR):
        return []

    pattern = os.path.join(CLAUDE_DIR, "**", "*.jsonl")
    files = glob.glob(pattern, recursive=True)

    # Filter by project name if specified
    if project:
        files = [f for f in files if project.lower() in f.lower()]

    # Sort by modification time, newest first
    files.sort(key=lambda f: os.path.getmtime(f), reverse=True)

    if limit:
        files = files[:limit]

    return files


# ─── Prefill ─────────────────────────────────────────────────────────────────

def prefill_claudeai(export_path: str, limit: int = None, dry_run: bool = False):
    """Prefill from a Claude.ai conversations.json export."""
    conversations = read_claudeai_export(export_path, limit)

    print(f"\n{BOLD}CURATOR — Prefill from Claude.ai Export{RESET}")
    print(f"{DIM}Found {len(conversations)} conversation(s) with user messages{RESET}")
    if dry_run:
        print(f"{YELLOW}DRY RUN — nothing will be stored{RESET}")
    print()

    conn = None if dry_run else connect(DB_PATH)

    total_messages = 0
    total_conceptions = 0
    skipped = 0

    for i, (name, texts) in enumerate(conversations):
        print(f"{DIM}[{i+1}/{len(conversations)}] {name[:60]} — {len(texts)} message(s){RESET}")

    for i, (name, texts) in enumerate(conversations):
        # Filter before batching
        clean = [t for t in texts if len(t.strip()) >= 15 and not t.strip().startswith("/")]
        skipped += len(texts) - len(clean)

        if not clean:
            continue

        print(f"{DIM}[{i+1}/{len(conversations)}] {name[:60]} — {len(clean)} message(s){RESET}")

        if dry_run:
            for text in clean:
                from observe import evaluate_signal_quality
                sq = evaluate_signal_quality(text)
                marker = f"{GREEN}●{RESET}" if sq.score >= 0.5 else f"{DIM}○{RESET}"
                print(f"  {marker} {DIM}[{sq.score:.2f}]{RESET} {text[:80]}")
        else:
            result = batch_observe(conn, clean, source=f"claudeai:{name[:30]}")
            total_conceptions += result["created"]
            skipped += result["skipped"]
            total_messages += len(clean)
            if result["created"] > 0:
                print(f"  {GREEN}+{result['created']} conceptions{RESET}")

    print()
    print(f"{BOLD}Done.{RESET}")
    print(f"{DIM}Messages processed: {total_messages}{RESET}")
    print(f"{DIM}Messages skipped: {skipped}{RESET}")

    if not dry_run:
        print(f"{DIM}Conceptions created: {total_conceptions}{RESET}")
        surfaced = surface(conn, SignalQuality(score=0.9, reason="prefill"), limit=10)
        if surfaced:
            print(f"\n{BOLD}Top conceptions now in space:{RESET}")
            for c in surfaced:
                print(f"  {DIM}rec={c.recency:.2f} conf={c.confidence:.2f}{RESET} {c.content[:80]}")
    print()


def prefill(project: str = None, limit: int = None, dry_run: bool = False):
    files = find_jsonl_files(project, limit)

    if not files:
        print(f"{YELLOW}No JSONL files found in {CLAUDE_DIR}{RESET}")
        if project:
            print(f"{DIM}Project filter: '{project}'{RESET}")
        return

    print(f"\n{BOLD}CURATOR — Prefill from Claude Code History{RESET}")
    print(f"{DIM}Reading {len(files)} conversation file(s){RESET}")
    if dry_run:
        print(f"{YELLOW}DRY RUN — nothing will be stored{RESET}")
    print()

    conn = None if dry_run else connect(DB_PATH)

    total_messages = 0
    total_conceptions = 0
    skipped = 0

    for i, filepath in enumerate(files):
        project_name = filepath.replace(CLAUDE_DIR, "").split(os.sep)[1] if CLAUDE_DIR in filepath else "unknown"
        filename = os.path.basename(filepath)

        messages = read_jsonl(filepath)
        user_messages = extract_user_messages(messages)

        if not user_messages:
            continue

        print(f"{DIM}[{i+1}/{len(files)}] {project_name}/{filename} — {len(user_messages)} user messages{RESET}")

        for text in user_messages:
            # Skip very short inputs — unlikely to contain useful conceptions
            if len(text.strip()) < 15:
                skipped += 1
                continue

            # Skip slash commands
            if text.strip().startswith("/"):
                skipped += 1
                continue

            total_messages += 1

            if dry_run:
                # Just evaluate signal quality, don't store
                sq = evaluate_signal_quality(text)
                if sq.score >= 0.5:
                    print(f"  {GREEN}●{RESET} {DIM}[{sq.score:.2f}]{RESET} {text[:80]}")
                else:
                    print(f"  {DIM}○ [{sq.score:.2f}] {text[:80]}{RESET}")
            else:
                result = observe(conn, text, source=f"prefill:{project_name}")
                conceptions_created = sum(
                    1 for a in result["actions"]
                    if a["action"] in ("created", "competing_conception_created")
                )
                total_conceptions += conceptions_created

                if conceptions_created > 0:
                    sq = result["signal_quality"]
                    print(f"  {GREEN}+{conceptions_created}{RESET} {DIM}[{sq['score']:.2f}] {text[:70]}{RESET}")

            # Small delay to avoid hammering the API
            time.sleep(0.2)

    print()
    print(f"{BOLD}Done.{RESET}")
    print(f"{DIM}Messages processed: {total_messages}{RESET}")
    print(f"{DIM}Messages skipped (short/commands): {skipped}{RESET}")

    if not dry_run:
        print(f"{DIM}Conceptions created: {total_conceptions}{RESET}")

        # Show what's now in the conception space
        surfaced = surface(conn, SignalQuality(score=0.9, reason="prefill"), limit=10)
        if surfaced:
            print(f"\n{BOLD}Top conceptions now in space:{RESET}")
            for c in surfaced:
                print(f"  {DIM}rec={c.recency:.2f} conf={c.confidence:.2f}{RESET} {c.content[:80]}")
    print()


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prefill Curator conception space from conversation history"
    )
    parser.add_argument(
        "--claudeai", "-c",
        help="Path to Claude.ai conversations.json export",
        default=None,
        metavar="FILE"
    )
    parser.add_argument(
        "--reset", "-r",
        help="Clear all conceptions before prefilling (ignored in dry-run)",
        action="store_true"
    )
    parser.add_argument(
        "--project", "-p",
        help="Filter by project name (partial match)",
        default=None
    )
    parser.add_argument(
        "--limit", "-l",
        help="Max number of conversation files to process (newest first)",
        type=int,
        default=None
    )
    parser.add_argument(
        "--dry-run", "-d",
        help="Show what would be extracted without storing",
        action="store_true"
    )

    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY") and not args.dry_run:
        print(f"\033[91mError: ANTHROPIC_API_KEY not set{RESET}")
        print(f"{DIM}export ANTHROPIC_API_KEY=your_key_here{RESET}")
        print(f"{DIM}Or use --dry-run to preview without storing{RESET}")
        sys.exit(1)

    if args.reset and not args.dry_run:
        conn = connect(DB_PATH)
        conn.execute("DELETE FROM conceptions")
        conn.execute("DELETE FROM observations")
        try:
            conn.execute("DELETE FROM conception_embeddings")
        except Exception:
            pass
        conn.commit()
        print(f"{YELLOW}Conception space cleared.{RESET}\n")

    if args.claudeai:
        prefill_claudeai(
            export_path=args.claudeai,
            limit=args.limit,
            dry_run=args.dry_run
        )
    else:
        prefill(
            project=args.project,
            limit=args.limit,
            dry_run=args.dry_run
        )
