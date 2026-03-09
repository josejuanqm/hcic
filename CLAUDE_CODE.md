# Curator — Claude Code Integration

Replaces static CLAUDE.md memory with a live weight-based conception space.
No separate API key needed — you are the intelligence layer. The MCP server is pure Weight engine.

## Your role

You handle all semantic decisions:
- Is this signal clear or ambiguous?
- What atomic conceptions should be extracted?
- Does this confirm or contradict an existing conception?

The Curator handles all weight mechanics:
- Storing conceptions with recency and confidence
- Decay over time (rate proportional to confidence)
- Surfacing by threshold

## Session start

Always call `surface()` first. Use the returned conceptions as your active context.

## When the user shares context

1. Call `find_related(content)` to check for existing conceptions
2. Decide: confirming, contradicting, or new?
   - **Confirming** → `update_weight(id, +0.05 to +0.3)`
   - **Contradicting** → `update_weight(id, -0.1 to -0.5)` then `create_conception(new_content)`
     - Do NOT delete the existing one. Contradictions coexist at different confidence levels.
   - **New** → `create_conception(content, initial_confidence=0.1)`
   - **Explicit correction** ("no, actually..." / "that's wrong") → `update_weight(id, -0.8)` then `create_conception(new_content, initial_confidence=0.4)`

## Signal quality judgment

Before extracting conceptions, judge signal quality yourself:
- "yeah that" → low (0.1) — extract nothing
- "I prefer dark mode" → high (0.85) — extract
- "I changed my mind about the deadline" → medium (0.7) — extract cautiously

Don't extract conceptions from ambiguous or context-dependent input.

## What makes a good conception

- Self-contained: "User prefers 2-space indentation" not "prefers spaces"
- Specific: "User is building Krill, a Docker management app in Swift" not "user builds apps"
- Third-person: always phrased as a statement about the user

## Setup

Edit `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "curator": {
      "command": "/path/to/hcic/.venv/bin/python3",
      "args": ["/path/to/hcic/curator/mcp_server.py"]
    }
  }
}
```

No API key needed. Restart Claude Code after adding.
