# Multi-Agent Coordination Pattern

## Problem

Single agents hit limits:
- **Context limits** — Can't hold full research + implementation in one session
- **Specialization** — Need separate researchers, implementers, reviewers
- **Parallelism** — Want to explore multiple approaches simultaneously

But multi-agent systems introduce chaos:
- Workers duplicate each other's research
- Coordinators delegate understanding instead of synthesizing
- Context inheritance explodes exponentially

## Golden Rules

### The Coordinator Must Synthesize, Not Delegate Understanding

**Anti-pattern:**
> "Based on your findings, fix the authentication system."

**Pattern:**
> "Research identified 3 auth flows: login, logout, token refresh. Implement ONLY the token refresh handler using the JWT strategy documented in [research output]. Return: implementation diff + test results."

The coordinator (orchestrator) adds value by digesting worker results into precise specs before dispatching implementation.

### Three Delegation Patterns

| Pattern | Context Sharing | Best For | Constraints |
|---------|----------------|----------|-------------|
| **Coordinator** | None — workers start fresh | Complex multi-phase tasks (research → synthesize → implement → verify) | Slowest but safest |
| **Fork** | Full — child inherits parent history | Quick parallel splits sharing loaded context | **Single-level only** — recursive forks multiply context cost |
| **Swarm** | Peer-to-peer via shared task list | Long-running independent workstreams | **Flat roster** — teammates can't spawn other teammates |

### Results Arrive Asynchronously; Fire-and-Forget Registration Returns ID Immediately

```typescript
// Example: Spawn worker, get ID back immediately
const taskId = await coordinator.spawn({
  type: 'research',
  prompt: 'Analyze auth flows...',
  toolFilter: ['read', 'search'], // Restrict tools
});

// Parent can continue working while worker runs
// Results arrive via callback or polling
```

## When To Use

- Task too large for single agent session
- Need parallel exploration (e.g., prototype multiple approaches)
- Want persistent specialized teammates (researcher, implementer, reviewer)
- Complex multi-phase workflows

## Tradeoffs

| Pattern | Speed | Safety | Context Cost |
|---------|-------|--------|--------------|
| **Coordinator** | Slowest | Safest | Lowest (zero inheritance) |
| **Fork** | Fastest | Medium | Highest (full inheritance) |
| **Swarm** | Medium | Medium | Medium (shared state only) |

## Implementation Patterns

### Coordinator Pattern (Recommended for Complex Tasks)

Phased workflow:

```
Phase 1: Research
  ↓ (synthesize findings)
Phase 2: Plan  
  ↓ (precise specs)
Phase 3: Implement
  ↓ (verify)
Phase 4: Review
```

```typescript
// Example: Coordinator workflow
const research = await coordinator.spawn({
  role: 'researcher',
  prompt: `Analyze existing authentication in ${authDir}.
  Find: login flow, logout flow, token handling.
  Return: structured findings only. NO implementation suggestions.`,
  toolFilter: ['read', 'search', 'glob'], // Can't write
});

await coordinator.synthesize(research.results);

const implement = await coordinator.spawn({
  role: 'implementer',
  prompt: `Implement token refresh handler using the JWT strategy
  from [Phase 2 findings]. 
  Constraints: Use existing AuthService patterns, add tests.`,
  toolFilter: ['read', 'search', 'edit', 'test'], // Can write
});
```

### Fork Pattern (Single-Level Only)

```typescript
// Parent spawns children for parallel work
const forks = await Promise.all([
  coordinator.fork({
    prompt: 'Implement login handler',
    inheritContext: true, // Full parent history
  }),
  coordinator.fork({
    prompt: 'Implement logout handler',
    inheritContext: true,
  }),
]);

// CRITICAL: Children must not fork recursively
// If allowed, context cost multiplies: parent + child1 + child2 + ...
```

### Swarm Pattern (Flat Roster)

```typescript
// Swarm: persistent team with shared task list
const swarm = new Swarm([
  { id: 'researcher', specialty: 'research' },
  { id: 'implementer', specialty: 'implementation' },
  { id: 'reviewer', specialty: 'verification' },
]);

// Agents pick tasks from shared queue
// Results posted back to shared state
await swarm.dispatch({
  taskId: 'feat-001',
  pickedBy: 'implementer',
});
```

## Gotchas

1. **Fork children must not fork** — Recursive guard preserves single-level invariant. Keep fork tool in child's pool (for prompt cache sharing) but block at call time.
2. **Coordinator workers start with zero context** — Only explicit prompt is passed. Don't assume child sees parent's accumulated research.
3. **Swarm teammates cannot spawn other teammates** — Roster is flat to prevent uncontrolled growth.
4. **Write self-contained prompts** — "Based on your findings" is an anti-pattern. Coordinator must digest first.
5. **Filter each worker's tool set** — Researcher doesn't need write; implementer doesn't need broad search.

## Related Patterns

- [Context Engineering](context-engineering-pattern.md) — Isolation patterns for delegation
- [Lifecycle & Bootstrap](lifecycle-bootstrap-pattern.md) — How agents are spawned at init

## Template: Worker Prompt Structure

```markdown
# Self-Contained Worker Prompt

## Context (Copied from Coordinator Synthesis)

**Task**: Implement token refresh handler
**Background**: Research identified JWT-based auth with 24h access tokens.
**Decision**: Use refresh token rotation (new refresh token on each refresh).

## Your Role

You are an **implementer**. Your job is to write production code following the specs above.

## Constraints

- Use existing patterns from `${authServicePath}`
- Add tests for success and failure cases
- Do NOT modify login/logout handlers (separate task)

## Your Tools

- read, search, edit, test
- Shell: npm test, npm run check only

## Deliverable

Return:
1. Implementation diff (files changed)
2. Test results (pass/fail)
3. Any blockers or clarifications needed

**Do NOT return**: Research findings, architectural debates, alternative designs.
```

## Evidence

Multi-agent coordination patterns are observed in production systems where:
- Coordinator workers start with zero context inheritance
- Fork is restricted to single-level to control context explosion
- Swarm agents communicate through shared task lists, not direct prompts
- Results arrive asynchronously with fire-and-forget registration
