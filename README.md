# hcic — Human Connection as Intelligent Context

A proof-of-concept implementation of the abstraction proposed in the paper *Human Connection as Architecture: A New Model for Persistent Intelligent Context*.

## The Idea

Every major memory system for AI — Mem0, MemGPT, LangChain, Apple Intelligence, Claude's own memory — shares a single foundational assumption: the consumer of data knows what it wants before retrieval begins. We call this the **intent-first assumption**.

This project proposes and implements a different abstraction, modeled on how human connection actually works: accumulated understanding expressed as anticipation, not storage and retrieval.

## Three Primitives

**Observe** — holistic, passive, continuous. Everything that passes through an interaction is automatically a candidate for future relevance. Nothing is explicitly saved.

**Weight** — two-dimensional property governing all conceptions:
- *Recency*: governs the present. Starts high, decays at a rate proportional to confidence.
- *Confidence*: governs persistence. Grows through confirming signal, shrinks through contradiction. Contradictions coexist as competing conceptions — they don't overwrite.

**Surface** — activates relevant context at the moment needed, gated by Signal Quality. Low signal quality surfaces the ambiguity itself rather than resolving it.

## Structure

```
curator/        # Core engine — schema, Weight mechanics, Surface
docs/           # Paper and supporting writing
```

## Status

Active development. Schema and Weight engine complete. Observe layer and agent wiring in progress.

## Paper

*Human Connection as Architecture: A New Model for Persistent Intelligent Context*  
Jose Quintero · Claude (Anthropic) — March 2026

Available in `docs/`.
