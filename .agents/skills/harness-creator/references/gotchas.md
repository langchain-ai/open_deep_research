# Gotchas — Harness Engineering Failure Modes

Non-obvious principles that will cause bugs if you violate them.

---

## 1. Memory Index Caps Fire Silently

**Symptom**: Recent memories "disappear" without error.

**Cause**: Index has hard caps (e.g., 200 lines / 25KB) enforced at read time. Long entries (multi-sentence summaries) hit byte cap while staying under line cap.

**Fix**: Keep index entries to one-line hooks. Put detail in topic files.

```markdown
✓ Good: "Use bun, not npm - user preference 2024-01-15"
✗ Bad: "The user prefers bun over npm because it's faster. This was discussed on 2024-01-15 when the user said 'use bun not npm' and I updated the package.json accordingly..."
```

---

## 2. Priority Ordering is Counterintuitive

**Symptom**: Global rule silently overridden by local file.

**Cause**: Local overrides beat project rules, which beat user rules, which beat org rules. If you inject at user level expecting it to dominate, a local override file in project root wins.

**Fix**: Test with full instruction-file stack present:

```bash
# Test priority ordering
cat ~/.claude/CLAUDE.md          # User level
cat ./CLAUDE.md                   # Project level  
cat ./CLAUDE.local.md             # Local override (WINS)
```

---

## 3. Extraction Timing Creates Race Window

**Symptom**: Background extractor writes memory, but user starts next turn before extraction completes.

**Cause**: Extraction fires at end of response. User can send message before extraction finishes.

**Fix**: Coalesce concurrent extraction requests. Advance cursor only after successful run. Failed extraction means those messages reconsidered next time.

---

## 4. Derivable Content Doesn't Belong in Memory

**Symptom**: Memory index fills with architecture details that stale quickly.

**Cause**: Agent saves what's derivable from codebase (architecture, code patterns, version history).

**Fix**: Exclude derivable content by design. Type taxonomy should forbid saving what's in the repo already.

---

## 5. Concurrent Classification is Per-Call, Not Per-Tool

**Symptom**: Tool marked "concurrent-safe" causes race conditions.

**Cause**: Same tool can be safe for some inputs and unsafe for others. Don't assume tool's concurrency behavior is static.

**Fix**: Classify each call at runtime:

```typescript
// Don't do this:
toolRegistry.register('shell', { concurrentSafe: false });

// Do this:
function isCallConcurrentSafe(call: ToolCall): boolean {
  if (call.args.command.startsWith('rm -rf')) return false;
  if (call.args.command.startsWith('cat')) return true;
  // ...runtime classification
}
```

---

## 6. Permission Evaluation Has Side Effects

**Symptom**: Permission check changes behavior on subsequent calls.

**Cause**: Permission evaluator tracks denials, transforms modes, updates state as side effect. Not a pure lookup function.

**Fix**: Don't cache permission results across calls. Re-evaluate each call fresh.

---

## 7. Most Async Work Skips "Pending" State

**Symptom**: UI shows "pending" but work unit never enters that state.

**Cause**: Work units register directly as "running" in practice. "Pending" exists in state machine but rarely used.

**Fix**: Don't build UI that assumes every work unit starts pending.

---

## 8. Fork Children Must Not Fork

**Symptom**: Context cost explodes exponentially.

**Cause**: Recursive forks multiply context: parent + child1 + child2 + grandchildren...

**Fix**: Enforce single-level invariant. Keep fork tool in child's pool (for prompt cache sharing) but block at call time.

---

## 9. Context Builders are Memoized but Manually Invalidated

**Symptom**: Model sees stale data for entire session.

**Cause**: Context builder cached at startup, but mutation doesn't clear cache.

**Fix**: Every mutation point must explicitly clear its corresponding cache:

```typescript
// Example: Cache invalidation at mutation point
async function editFile(path: string, content: string) {
  await writeFile(path, content);
  context.cache.invalidate(`file:${path}`); // MUST invalidate
}
```

---

## 10. Hook Trust is All-or-Nothing

**Symptom**: Entire extension system disabled because one hook untrusted.

**Cause**: If workspace untrusted, all hooks skip — not just suspicious ones.

**Fix**: Design hooks with trust gate at dispatch point. Don't attempt per-hook trust evaluation.

---

## 11. Eviction Requires Notification

**Symptom**: Parent can never read work unit result.

**Cause**: Work unit evicted before parent notified of completion. Race condition: parent tries to read result that's already GC'd.

**Fix**: Two-phase eviction:
1. Clean disk output at terminal state (eager)
2. Clean in-memory record after parent notified (lazy)

---

## 12. Skill Listing Budgets Are Tight

**Symptom**: Skill description truncated, can't trigger properly.

**Cause**: Skill descriptions concatenated and capped per entry (~150 chars). Front-loaded trigger language gets priority.

**Fix**: Front-load distinctive trigger language:

```markdown
✓ Good: "harness-patterns: Memory, permissions, context engineering, multi-agent"
✗ Bad: "A comprehensive skill for understanding and implementing various patterns related to AI agent harnesses and runtime systems..."
```

---

## 13. Default Tool Permission is "Allow"

**Symptom**: Tool bypasses expected gate.

**Cause**: Tools without custom permission logic delegate entirely to rule-based system. Default is "allow" unless configured otherwise.

**Fix**: Override default for sensitive tools:

```typescript
registry.register('shell', {
  defaultPermission: 'ask', // NOT 'allow'
  // ...
});
```

---

## 14. Team Memory Requires Auto-Memory Enabled

**Symptom**: Team-shared memory doesn't work even when configured.

**Cause**: Team memory builds on same directory/index infrastructure as auto-memory. Disabling auto-memory (via env var or settings) also disables team memory.

**Fix**: Ensure auto-memory enabled before enabling team memory. Check both feature gate and enablement check.

---

## 15. Orphaned Topic Files Accumulate

**Symptom**: Disk space fills with `.claude/memory/topics/` files.

**Cause**: Two-step save (topic file then index). Crash between steps leaves orphaned topic file.

**Fix**: Periodic sweep deletes topic files not referenced by index. Orphans don't corrupt index but consume disk space.

---

## Related Reading

- [Memory Persistence Pattern](memory-persistence-pattern.md) — Gotchas #1, #3, #4, #15
- [Tool Registry Pattern](tool-registry-pattern.md) — Gotchas #5, #6, #13
- [Multi-agent Pattern](multi-agent-pattern.md) — Gotchas #8, #11
- [Context Engineering Pattern](context-engineering-pattern.md) — Gotchas #9
- [Lifecycle Pattern](lifecycle-bootstrap-pattern.md) — Gotchas #10, #14
