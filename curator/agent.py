"""
Curator — Agent
A conversation loop where memory is managed by the Weight engine.
Context comes from Surface, not raw history injection.

Run: python3 curator/agent.py
Requires: ANTHROPIC_API_KEY
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(__file__))
from schema import connect, surface, SignalQuality
from observe import observe, evaluate_signal_quality

import anthropic

# ─── Colors ──────────────────────────────────────────────────────────────────

GREEN  = "\033[92m"
BLUE   = "\033[94m"
YELLOW = "\033[93m"
DIM    = "\033[2m"
BOLD   = "\033[1m"
RESET  = "\033[0m"
ORANGE = "\033[38;5;208m"

# ─── Generate ────────────────────────────────────────────────────────────────

client = anthropic.Anthropic()

def generate(user_msg: str, surfaced_conceptions: list) -> str:
    """
    Generate a response using surfaced conceptions as context.
    Not raw history. Not a memory file. Just what the Weight engine decided matters now.
    """
    if surfaced_conceptions:
        context = "\n".join(
            f"- {c.content} [confidence: {c.confidence:.2f}, recency: {c.recency:.2f}]"
            for c in surfaced_conceptions
        )
        system = f"""You are a conversational agent. Your memory is managed by the Curator — 
a weight-based context system that surfaces what is currently relevant based on accumulated signal.

Active context (surfaced from conception space, ordered by relevance):
{context}

Use this context naturally in your response. Do not announce or reference the memory system.
Be conversational and direct."""
    else:
        system = """You are a conversational agent with no accumulated context yet. 
Respond naturally. Context will build as the conversation develops."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=600,
        system=system,
        messages=[{"role": "user", "content": user_msg}]
    )
    return response.content[0].text


# ─── Display ─────────────────────────────────────────────────────────────────

def print_conception_space(conceptions: list, surfaced_ids: set):
    if not conceptions:
        return
    print(f"\n{DIM}{'─' * 60}{RESET}")
    print(f"{DIM}  Conception space ({len(conceptions)} active){RESET}")
    for c in conceptions[:8]:
        surfaced = c.id in surfaced_ids
        marker = f"{GREEN}●{RESET}" if surfaced else f"{DIM}○{RESET}"
        group = f" {ORANGE}[competing]{RESET}" if c.groupId else ""
        bar_rec = int(c.recency * 20)
        bar_conf = int(c.confidence * 20)
        rec_bar = f"{YELLOW}{'█' * bar_rec}{'░' * (20 - bar_rec)}{RESET}"
        conf_bar = f"{BLUE}{'█' * bar_conf}{'░' * (20 - bar_conf)}{RESET}"
        content_short = c.content[:55] + ("…" if len(c.content) > 55 else "")
        print(f"  {marker} {DIM}{content_short}{RESET}{group}")
        print(f"    rec {rec_bar} {c.recency:.2f}  conf {conf_bar} {c.confidence:.2f}")
    print(f"{DIM}{'─' * 60}{RESET}\n")


def print_observe_summary(result: dict):
    sq = result["signal_quality"]
    score = sq["score"]
    color = GREEN if score > 0.6 else YELLOW if score > 0.3 else "\033[91m"
    print(f"{DIM}  signal {color}{score:.2f}{RESET}{DIM} — {sq['reason']}{RESET}")

    for action in result["actions"]:
        a = action["action"]
        if a == "created":
            print(f"{DIM}  + created #{action['id']}{RESET}")
        elif a == "confirmed":
            print(f"{DIM}  ↑ confirmed #{action['id']} (+{action['delta']:.2f}){RESET}")
        elif a == "competing_conception_created":
            explicit = " [explicit]" if action.get("explicit_instruction") else ""
            print(f"{DIM}  ⇌ competing: weakened #{action['existing_id']}, created #{action['new_id']}{explicit}{RESET}")


# ─── Main Loop ───────────────────────────────────────────────────────────────

def run():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(f"\033[91mError: ANTHROPIC_API_KEY not set\033[0m")
        sys.exit(1)

    conn = connect("curator_agent.db")  # persists between sessions

    print(f"\n{BOLD}CURATOR AGENT{RESET}")
    print(f"{DIM}Memory is managed by the Weight engine. Context comes from Surface.{RESET}")
    print(f"{DIM}Type 'quit' to exit. Type 'reset' to clear conception space.{RESET}\n")

    while True:
        try:
            user_input = input(f"{BOLD}You:{RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            break

        if user_input.lower() == "reset":
            conn.execute("DELETE FROM conceptions")
            conn.execute("DELETE FROM conception_embeddings")
            conn.execute("DELETE FROM observations")
            conn.commit()
            print(f"{DIM}  Conception space cleared.{RESET}\n")
            continue

        if user_input.lower() == "inspect":
            from schema import getLiveConceptions
            rows = conn.execute(
                "SELECT id, content, recency, confidence, last_updated FROM conceptions ORDER BY recency DESC"
            ).fetchall()
            print(f"\n{DIM}All conceptions:{RESET}")
            for r in rows:
                print(f"  #{r[0]} rec={r[1]:.3f} conf={r[2]:.3f} | {r[3][:70]}")
            print()
            continue

        # 1. Observe — update conception space
        obs_result = observe(conn, user_input)
        print_observe_summary(obs_result)

        # 2. Surface — what's relevant right now
        sq_score = obs_result["signal_quality"]["score"]
        sq = SignalQuality(score=sq_score, reason=obs_result["signal_quality"]["reason"])
        surfaced = surface(conn, sq)
        surfaced_ids = {c.id for c in surfaced}

        if surfaced:
            print(f"{DIM}  surfaced {len(surfaced)} conception{'s' if len(surfaced) != 1 else ''}{RESET}")

        # 3. Generate response using surfaced context
        response = generate(user_input, surfaced)

        print(f"\n{BOLD}Curator:{RESET} {response}\n")

        # 4. Show conception space state
        all_conceptions = conn.execute(
            "SELECT id, content, recency, confidence, last_updated FROM conceptions ORDER BY recency DESC LIMIT 8"
        ).fetchall()

        class C:
            def __init__(self, row):
                from schema import _compute_current_recency
                import time
                self.id = row[0]
                self.content = row[3]
                self.recency = _compute_current_recency(row[1], row[2], row[4])
                self.confidence = row[2]
                self.groupId = None

        live = [C(r) for r in all_conceptions]

        # Mark competing conceptions
        group_ids = conn.execute(
            "SELECT id, content FROM conceptions WHERE id IN (SELECT DISTINCT conception_id FROM conception_embeddings)"
        ).fetchall()

        print_conception_space(live, surfaced_ids)


if __name__ == "__main__":
    run()
