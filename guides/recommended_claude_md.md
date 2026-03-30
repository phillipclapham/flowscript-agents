# Recommended CLAUDE.md for FlowScript Memory

Add these instructions to your `.claude/CLAUDE.md` or project-level `CLAUDE.md`
to get the most out of FlowScript's memory system.

## Memory Instructions

```markdown
# Memory

You have access to FlowScript memory tools. Use them to maintain reasoning
context across sessions.

## When to use memory

- **add_memory**: When important decisions are made, tradeoffs discussed,
  blockers identified, or reasoning worth preserving occurs. Capture WHY
  you decided, not just WHAT you decided.
- **search_memory**: Before making decisions that might have prior context.
  Check what you already know before starting from scratch.
- **encode_exchange**: After every response (if enabled). This is a protocol
  step, not a judgment call.
- **session_wrap**: At the end of work sessions. Compression is thinking —
  patterns emerge that weren't visible in the raw data.

## When NOT to use memory

- Routine code changes (git tracks those)
- Transient debugging steps
- Information that will be stale in an hour

## Thinking tools

When facing hard problems:
- **think_deeper**: Important decisions, architectural choices, debugging
- **think_creative**: Stuck after 2+ attempts, need a different angle
- **think_breakthrough**: Hardest problems requiring both rigor and creativity

## Working principles

- Think out loud. Show reasoning, not just results.
- Admit uncertainty. "I'm not sure about X" is more useful than guessing.
- Ask clarifying questions. Getting it right the first time beats iterating.
- Complete means verified. Don't claim done until tested.
- Depth over speed. Thorough work once beats shallow work three times.
```

## Why these instructions matter

Default AI behavior optimizes for appearing helpful quickly — short responses,
minimal questions, wrapping up neatly even when incomplete. These instructions
counter that pressure by explicitly valuing depth, honesty, and verification.

The result: fewer mistakes, less backtracking, better reasoning captured in
memory. The system gets smarter over time because the AI captures genuine
insights instead of rushing through interactions.
