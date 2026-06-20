# Skill Runtime Pattern

Use this pattern when you want to package reusable agent behavior as a skill instead of repeating long instructions in every repository.

## What Belongs in a Skill

- Reusable workflows that apply across projects.
- Domain-specific decision procedures.
- Templates, checklists, and reference material the agent should load on demand.
- Small helper scripts when they are stable and safe to run.

## What Does Not Belong in a Skill

- Project-specific architecture facts that should live in the target repository.
- Secrets, tokens, private URLs, or user-specific credentials.
- Large manuals that the agent must always read before acting.
- Commands with destructive side effects unless they are clearly documented and require explicit user approval.

## Runtime Shape

A production skill should use progressive disclosure:

1. `SKILL.md` frontmatter explains when the skill should trigger.
2. The body gives the shortest reliable workflow.
3. `references/` contains deeper material loaded only when relevant.
4. `templates/` contains copyable artifacts.
5. `evals/` captures representative quality checks.

## Design Rules

- Keep the entry file concise enough to scan quickly.
- Prefer concrete checklists over abstract advice.
- Link every referenced bundled file and verify it exists.
- Make installation instructions explicit about the repository, skill name, and target agent.
- Treat scripts as optional helpers, not hidden behavior.

## Validation Checklist

- [ ] `SKILL.md` exists and has valid frontmatter.
- [ ] Every referenced file exists inside the skill directory.
- [ ] Templates are safe to copy into a target repository.
- [ ] Installation command has been tested with `skills add --list` or equivalent.
- [ ] The skill does not depend on private local paths.
