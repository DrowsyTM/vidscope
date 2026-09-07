# Contributing Guide

Thank you for your interest in contributing. This guide records the
project's contribution expectations without prescribing a particular toolchain.

## Getting Started

Read the project documentation and existing contribution guidance before
making changes. Confirm the supported runtime, dependency setup, and local
verification commands from the repository's own configuration.

## Quality Expectations

Keep changes focused, explain user-visible behavior, and add or update tests
for changed behavior. Run the repository's documented checks before opening a
pull request. Keep diagnostic output out of normal command output when the
project distinguishes logs from results.

## Changelog Policy

`CHANGELOG.md` is maintained manually. The release workflow packages and publishes distributions but does not generate changelog entries.

### When to add an entry

Add a concise bullet under `## [Unreleased]` for changes that affect users or operators, including:

- New or changed CLI, MCP, or API behavior.
- Meaningful bug fixes.
- Breaking changes or deprecations.
- Security fixes.
- Dependency, runtime, deployment, or configuration changes that affect supported usage.

Test-only changes, internal refactors, formatting-only changes, and documentation-only changes without user-visible impact do not require an entry.

### Entry format

- Use the existing Keep a Changelog categories: `Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`, and `Security`.
- Describe the observable change and its impact; omit implementation details.
- Keep one logical change per concise bullet and match the existing changelog style.
- Keep version and date headings unchanged; create or update them only during release preparation.
- In the pull request description, mention the changelog entry or explain why no entry applies.

## Making Changes

1. Create a focused branch or change set according to the project's workflow.
2. Write tests that demonstrate the intended behavior.
3. Review the complete diff, including configuration and generated files.
4. Update relevant documentation and record user-impacting changes according to the Changelog Policy.
5. Open a pull request using the repository's documented process.
