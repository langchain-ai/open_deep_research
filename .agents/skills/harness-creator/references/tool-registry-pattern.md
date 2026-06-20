# Tool Registry and Safety Pattern

## Problem

Agents need tools (shell, file edit, search, etc.) to be productive. But unbounded tool access creates risks:

- Destructive operations (rm -rf, DROP TABLE, etc.)
- Race conditions from concurrent tool calls
- Silent policy violations from misconfigured permissions

The solution is a **fail-closed registry** with explicit concurrency classification and a multi-source permission pipeline.

## Golden Rules

### Default to Fail-Closed

Tools are **non-concurrent** and **non-read-only** unless explicitly marked safe. This prevents:
- Accidental parallel execution of state-mutating operations
- Silent data corruption from concurrent writes

### Concurrency is Per-Call, Not Per-Tool

The same tool can be safe for some inputs and unsafe for others:

```
✓ Safe (can run in parallel):
  - cat file1.txt
  - grep "pattern" src/
  - ls -la

✗ Unsafe (must run serially):
  - rm -rf build/
  - npm install (network, filesystem mutation)
  - sed -i 's/old/new/g' *.ts
```

The runtime partitions a batch of tool calls into consecutive groups: safe calls run in parallel; any unsafe call starts a serial segment.

### Permission Pipeline has Side Effects

The permission evaluator is **stateful** — it:
- Tracks denials (for audit and rate limiting)
- Transforms modes (e.g., auto → ask after denial)
- Updates session state as a side effect

**Strict priority order:**
```
Policy (org-wide) → User settings → Project rules → Local overrides → Session grants
```

## When To Use

- Your agent runtime needs tool registration
- You need concurrency control for parallel tool calls
- You need permission gating (auto-approve, ask-first, deny)
- You need to track tool usage for audit

## Tradeoffs

| Decision | Benefit | Cost |
|---|---|---|
| Fail-closed defaults | New tools are safe out of the box | Developers must actively opt into concurrency |
| Per-call classification | Fine-grained control over parallelism | Requires analyzing each call, not just tool registration |
| Multi-source permission layering | Flexible policy composition | Hard to debug when rules conflict |
| Stateful evaluator | Can adapt behavior based on history | Not a pure function — harder to test |

## Implementation Patterns

### Tool Registration

```typescript
// Example: Tool registry entry
interface ToolDefinition {
  name: string;
  description: string;
  handler: (args: any) => Promise<any>;
  
  // Safety classification
  isReadOnly: boolean;       // Default: false
  isConcurrentSafe: boolean; // Default: false
  
  // Optional custom permission logic
  permissionCheck?: (args: any, context: ToolContext) => PermissionResult;
}

// Register tools
registry.register('read_file', {
  name: 'read_file',
  description: 'Read contents of a file',
  handler: readFile,
  isReadOnly: true,
  isConcurrentSafe: true,  // Safe to read multiple files in parallel
});

registry.register('write_file', {
  name: 'write_file',
  description: 'Write or overwrite a file',
  handler: writeFile,
  isReadOnly: false,
  isConcurrentSafe: false, // Must run serially to prevent race conditions
});
```

### Permission Pipeline

```typescript
// Permission evaluation order
async function evaluatePermission(
  toolCall: ToolCall,
  context: PermissionContext
): Promise<PermissionResult> {
  
  // 1. Policy rules (highest priority, org-wide)
  const policyResult = await policyEngine.check(toolCall, context);
  if (policyResult !== 'defer') return policyResult;
  
  // 2. User settings
  const userResult = await userSettings.check(toolCall, context);
  if (userResult !== 'defer') return userResult;
  
  // 3. Project rules
  const projectResult = await projectRules.check(toolCall, context);
  if (projectResult !== 'defer') return projectResult;
  
  // 4. Local overrides
  const localResult = await localOverrides.check(toolCall, context);
  if (localResult !== 'defer') return localResult;
  
  // 5. Session grants (lowest priority)
  return sessionGrants.check(toolCall, context);
}
```

### Bypass-Immune Rules

Certain paths or operations should never be auto-approved:

```yaml
# Protected paths (never auto-approve)
protected_paths:
  - /etc/**
  - /usr/**
  - node_modules/**
  - .git/**

# Protected commands (always ask)
protected_commands:
  - "rm -rf*"
  - "DROP TABLE*"
  - "DELETE FROM*"
  - "mkfs*"
```

## Gotchas

1. **Most async work skips "pending" state** — work units register directly as "running"
2. **Permission evaluation has side effects** — don't cache results across calls
3. **Concurrency classification requires analyzing inputs**, not just tool name
4. **The default permission for tools is "allow"** — tools without custom logic delegate to rule-based system
5. **Eviction requires notification** — terminal work units only GC-eligible after parent notified

## Related Patterns

- [Lifecycle & Bootstrap](lifecycle-bootstrap-pattern.md) — How tools are registered at init
- Hook Lifecycle (`hook-lifecycle-pattern.md`) — Pre/post tool execution hooks

## Template: Tool Safety Checklist

Before enabling a new tool:

```markdown
## Tool Safety Review

**Tool name**: [e.g., execute_shell]

### Classification
- [ ] Determined if read-only (true / false / depends on args)
- [ ] Determined if concurrent-safe (true / false / depends on args)
- [ ] Documented unsafe input patterns

### Permission Requirements
- [ ] Default mode set to "ask" or "deny"
- [ ] Bypass-immune paths/commands defined
- [ ] Custom permission logic implemented (if needed)
- [ ] Audit logging enabled

### Testing
- [ ] Tested with safe inputs (should auto-approve)
- [ ] Tested with unsafe inputs (should ask/deny)
- [ ] Tested concurrent execution (should serialize if unsafe)
- [ ] Tested error handling (failures logged, state consistent)
```

## Evidence

Tool registry and safety patterns are observed in production agent runtimes including:
- Claude Code's tool registry with explicit concurrency flags
- Multi-source permission evaluation (settings → project → session)
- Protected path/command lists that bypass auto-approve modes
- Per-call concurrency classification that partitions tool batches
