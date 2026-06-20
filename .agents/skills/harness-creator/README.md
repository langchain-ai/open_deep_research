# harness-creator

A compact skill for building and auditing harnesses around AI coding agents.

It helps a repository provide five things agents need: instructions, state, verification, scope boundaries, and lifecycle handoff.

## Install

```bash
npx skills add walkinglabs/learn-harness-engineering --skill harness-creator
```

Or copy `skills/harness-creator/` into your skill path.

## Use

```bash
node skills/harness-creator/scripts/create-harness.mjs --target /path/to/project
node skills/harness-creator/scripts/validate-harness.mjs --target /path/to/project
node skills/harness-creator/scripts/run-benchmark.mjs --target /path/to/project --html /path/to/report.html
```

The scripts use only Node.js built-in modules. They can be run after copying the skill directory into another repository.

## What It Creates

- `AGENTS.md` or `CLAUDE.md`
- `feature_list.json`
- `progress.md`
- `init.sh`
- `session-handoff.md`

`create-harness.mjs` detects common project types and package managers. It supports Node/npm/pnpm/yarn/bun, Python, Go, Rust, Maven, Gradle, and .NET at a basic verification-command level.

## What It Checks

`validate-harness.mjs` scores the five harness subsystems:

1. Instructions
2. State
3. Verification
4. Scope
5. Lifecycle

The score is structural. It tells you whether the harness is present and coherent; it does not replace real before/after agent-session testing.

## Status

- [x] Minimal harness scaffolding
- [x] Five-subsystem validation
- [x] HTML assessment report
- [x] Structural benchmark report
- [x] 10 eval cases
- [x] Generic verification detection for common stacks
- [ ] Optional real before/after agent-session replay

## Files

```text
harness-creator/
├── SKILL.md
├── metadata.json
├── agents/openai.yaml
├── scripts/
│   ├── create-harness.mjs
│   ├── validate-harness.mjs
│   ├── render-assessment-html.mjs
│   ├── run-benchmark.mjs
│   └── lib/harness-utils.mjs
├── templates/
│   ├── agents.md
│   ├── feature-list.json
│   ├── feature-list.schema.json
│   ├── init.sh
│   ├── progress.md
│   └── session-handoff.md
├── references/
└── evals/evals.json
```

## Boundaries

This skill is for harness engineering, not model selection, prompt tuning alone, or app architecture. Keep project-specific facts in the target repository.
