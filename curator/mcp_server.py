"""
Curator — MCP Server (no API calls)
Pure Weight engine. Claude Code is the intelligence layer.

Claude Code handles: signal quality, extraction, classification.
This server handles: weight math, decay, persistence, surface.

Tools:
  create_conception(content, initial_confidence?)
  update_weight(conception_id, delta)
  find_related(content) -> list of similar conceptions
  surface(signal_quality?, limit?) -> weighted conceptions
  inspect() -> full conception space

Setup in ~/.claude/settings.json:
  {
    "mcpServers": {
      "curator": {
        "command": "/path/to/hcic/.venv/bin/python3",
        "args": ["/path/to/hcic/curator/mcp_server.py"]
      }
    }
  }

No ANTHROPIC_API_KEY needed — Claude Code is already Claude.
"""

import os
import sys
import json
import asyncio

sys.path.insert(0, os.path.dirname(__file__))
from schema import (
    connect, surface as surface_fn, SignalQuality,
    create_conception, update_weight, find_related_conceptions,
    get_conception, _compute_current_recency,
    log_episode, find_related_episodes,
    INITIAL_CONFIDENCE
)
from observe import embed

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

# ─── DB ──────────────────────────────────────────────────────────────────────

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "curator_mcp.db")
conn = connect(DB_PATH)

server = Server("curator")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="create_conception",
            description="""Store a new conception in the weight engine.
Call this when you've determined a new atomic fact should be added to the conception space.
Start with low initial_confidence (default 0.1) unless this is an explicit user instruction
(use 0.4 for direct corrections like 'no, actually...' or 'I prefer...').""",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The conception as a self-contained statement. E.g. 'User prefers 2-space indentation'"
                    },
                    "initial_confidence": {
                        "type": "number",
                        "description": "Starting confidence 0.0-1.0. Default 0.1. Use 0.4 for explicit user instructions.",
                        "default": 0.1
                    }
                },
                "required": ["content"]
            }
        ),
        types.Tool(
            name="update_weight",
            description="""Update the weight of an existing conception.
Call with a positive delta for confirming signal (same thing said again, consistent behavior).
Call with a negative delta for contradicting signal (user said something different).
On contradiction: call update_weight(existing_id, negative_delta) to weaken it,
then create_conception() to add the competing conception alongside it.
Do NOT delete the existing conception — contradictions coexist.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "conception_id": {
                        "type": "integer",
                        "description": "ID of the conception to update (from find_related or surface)"
                    },
                    "delta": {
                        "type": "number",
                        "description": "Confidence change. Confirming: +0.05 to +0.3. Contradicting: -0.05 to -0.5. Explicit correction: -0.8."
                    }
                },
                "required": ["conception_id", "delta"]
            }
        ),
        types.Tool(
            name="find_related",
            description="""Find conceptions semantically related to a piece of text.
Call this before deciding whether to create a new conception or update an existing one.
Returns conceptions above similarity threshold with their current weights.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Text to find related conceptions for"
                    }
                },
                "required": ["content"]
            }
        ),
        types.Tool(
            name="surface",
            description="""Return currently relevant conceptions ordered by recency then confidence.
Call at the start of each session and before complex responses.
Recency governs the present (high recency = recently active).
Confidence governs persistence (high confidence = repeatedly confirmed).
Use returned context naturally — do not announce the memory system.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "signal_quality": {
                        "type": "number",
                        "description": "0.0-1.0. Use 0.9 for clear context, lower for ambiguous requests. Below 0.3 returns nothing.",
                        "default": 0.9
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max conceptions to return",
                        "default": 8
                    }
                }
            }
        ),
        types.Tool(
            name="log_episode",
            description="""Log a conversation exchange to the episode store.
Call this after EVERY user message — both the user input and a summary of your response.
This is separate from the conception space and never affects weights.
Used for episodic recall: "what did we work on last Tuesday?"

Call this unconditionally — every exchange, every session. It's a log, not a filter.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Stable identifier for this session (use current timestamp or project name)"
                    },
                    "user_input": {
                        "type": "string",
                        "description": "The user's exact message"
                    },
                    "assistant_summary": {
                        "type": "string",
                        "description": "1-3 sentence summary of what you did or said in response"
                    }
                },
                "required": ["session_id", "user_input", "assistant_summary"]
            }
        ),
        types.Tool(
            name="recall",
            description="""Search episode history for past conversations.
Use when the user asks about past work: "what did we do last week?", "did we solve X?", "what was the approach we used?"
Searches both user inputs and assistant summaries.
Returns chronological episodes matching the query.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for in past episodes"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max episodes to return",
                        "default": 10
                    }
                },
                "required": ["query"]
            }
        ),
        types.Tool(
            name="log_session",
            description="""Store a summary of the current session before it ends.
Call this when the user says goodbye, closes out, or when you sense the session is wrapping up.
Also call proactively after any significant decision, solution, or milestone.

Write a concise summary covering:
- What was worked on (project, feature, problem)
- Key decisions made
- Solutions found or approaches tried
- Anything the user should remember next session

This creates a high-recency conception that will surface next session.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "What happened this session. 2-5 sentences, specific and concrete."
                    }
                },
                "required": ["summary"]
            }
        ),
        types.Tool(
            name="inspect",
            description="Show full conception space with all weights. Use for debugging or when user asks what you remember.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:

    # ── Episode store (conversation log, separate from conception space) ──────

    if name == "log_episode":
        session_id = arguments.get("session_id", "unknown")
        user_input = arguments.get("user_input", "").strip()
        assistant_summary = arguments.get("assistant_summary", "").strip()

        if not user_input:
            return [types.TextContent(type="text", text="Error: user_input required")]

        # Embed the combined text for vector recall
        combined = f"{user_input} {assistant_summary}".strip()
        embedding = embed(combined)

        log_episode(conn, session_id, user_input, assistant_summary, embedding)
        return [types.TextContent(type="text", text="Episode logged.")]

    elif name == "recall":
        query = arguments.get("query", "").strip()
        limit = arguments.get("limit", 10)

        if not query:
            return [types.TextContent(type="text", text="Error: query required")]

        embedding = embed(query)
        episodes = find_related_episodes(conn, embedding, limit=limit)

        if not episodes:
            total = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
            return [types.TextContent(type="text", text=
                f"No related episodes found. Total logged: {total}"
            )]

        lines = [f"Episodes related to: '{query[:60]}'\n"]
        for ep in episodes:
            lines.append(
                f"[{ep['created_at'][:16]}] session:{ep['session_id'][:20]} "
                f"(similarity: {ep['similarity']:.2f})\n"
                f"  User: {ep['user_input'][:100]}\n"
                f"  {ep['assistant_summary'][:120] if ep['assistant_summary'] else '(no summary)'}"
            )
        return [types.TextContent(type="text", text="\n\n".join(lines))]

    # ── Conception space ──────────────────────────────────────────────────────

    elif name == "create_conception":
        content = arguments.get("content", "").strip()
        initial_confidence = arguments.get("initial_confidence", INITIAL_CONFIDENCE)

        if not content:
            return [types.TextContent(type="text", text="Error: content required")]

        embedding = embed(content)
        cid = create_conception(conn, content, embedding, "claude_code", initial_confidence)
        c = get_conception(conn, cid)

        return [types.TextContent(type="text", text=
            f"Created conception #{cid}\n"
            f"recency: {c.recency:.2f} | confidence: {c.confidence:.2f}\n"
            f"content: {content}"
        )]

    elif name == "update_weight":
        conception_id = arguments.get("conception_id")
        delta = arguments.get("delta", 0)

        if conception_id is None:
            return [types.TextContent(type="text", text="Error: conception_id required")]

        c_before = get_conception(conn, conception_id)
        if not c_before:
            return [types.TextContent(type="text", text=f"Error: conception #{conception_id} not found")]

        update_weight(conn, conception_id, delta)
        c_after = get_conception(conn, conception_id)

        direction = "↑" if delta > 0 else "↓"
        return [types.TextContent(type="text", text=
            f"Updated #{conception_id} {direction}\n"
            f"confidence: {c_before.confidence:.3f} → {c_after.confidence:.3f}\n"
            f"recency reset: {c_after.recency:.2f}"
        )]

    elif name == "find_related":
        content = arguments.get("content", "").strip()
        if not content:
            return [types.TextContent(type="text", text="Error: content required")]

        embedding = embed(content)
        related = find_related_conceptions(conn, embedding, limit=5)

        if not related:
            return [types.TextContent(type="text", text="No related conceptions found.")]

        lines = [f"Related conceptions for: '{content[:60]}'\n"]
        for cid, similarity in related:
            c = get_conception(conn, cid)
            if c:
                lines.append(
                    f"#{cid} (similarity: {similarity:.2f})\n"
                    f"  recency: {c.recency:.2f} | confidence: {c.confidence:.2f}\n"
                    f"  content: {c.content}"
                )

        return [types.TextContent(type="text", text="\n".join(lines))]

    elif name == "surface":
        signal_quality = arguments.get("signal_quality", 0.9)
        limit = arguments.get("limit", 8)

        sq = SignalQuality(score=signal_quality, reason="requested")
        conceptions = surface_fn(conn, sq, limit=limit)

        if not conceptions:
            total = conn.execute("SELECT COUNT(*) FROM conceptions").fetchone()[0]
            return [types.TextContent(type="text", text=
                f"No conceptions above threshold. Total in space: {total}"
            )]

        lines = ["Active context:\n"]
        for i, c in enumerate(conceptions, 1):
            lines.append(
                f"{i}. [{c.id}] {c.content}\n"
                f"   recency: {c.recency:.2f} | confidence: {c.confidence:.2f}"
            )

        total = conn.execute("SELECT COUNT(*) FROM conceptions").fetchone()[0]
        lines.append(f"\n{len(conceptions)} surfaced of {total} total")
        return [types.TextContent(type="text", text="\n".join(lines))]

    elif name == "log_session":
        summary = arguments.get("summary", "").strip()
        if not summary:
            return [types.TextContent(type="text", text="Error: summary required")]

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        content = f"[Session {timestamp}] {summary}"

        embedding = embed(content)
        cid = create_conception(conn, content, embedding, "session_log",
                                initial_confidence=0.15)
        conn.execute("UPDATE conceptions SET recency = 1.0 WHERE id = ?", (cid,))
        conn.commit()

        return [types.TextContent(type="text", text=
            f"Session logged #{cid}\n{content[:120]}"
        )]

    elif name == "inspect":
        rows = conn.execute(
            "SELECT id, content, recency, confidence, last_updated FROM conceptions ORDER BY recency DESC"
        ).fetchall()

        if not rows:
            return [types.TextContent(type="text", text="Conception space is empty.")]

        lines = [f"Conception space ({len(rows)} total):\n"]
        for row in rows:
            live_recency = _compute_current_recency(row[1], row[2], row[4])
            status = "active" if live_recency >= 0.15 else "faded"
            lines.append(
                f"#{row[0]} [{status}] rec={live_recency:.3f} conf={row[2]:.3f}\n"
                f"  {row[3][:80]}"
            )
        return [types.TextContent(type="text", text="\n".join(lines))]

    return [types.TextContent(type="text", text=f"Unknown tool: {name}")]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
