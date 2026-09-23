# Contributing

## Attribution

**Commits and pull request descriptions carry no AI attribution.** The history and the
PR record are the team's own work.

Concretely, none of these may appear in a commit message or a PR body:

```
Co-Authored-By: Claude <noreply@anthropic.com>
Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

This is enforced: the `attribution` workflow (`.github/workflows/attribution.yml`) fails a pull
request whose commits contain them.

This is about the record, not the tooling. Assistant tools remain welcome and the
`claude-review` workflow still runs on every PR — the point is only that the commit log
and PR descriptions read as authored by the people on the project.

### Turning it off in Claude Code

Claude Code adds these trailers by default. To stop it, add the instruction to your
`CLAUDE.md` or your user-level `~/.claude/CLAUDE.md`:

```
Never add Co-Authored-By trailers or any other AI attribution to commits or PR
descriptions.
```

A project-level `CLAUDE.md` applies to everyone who clones the repo; a user-level one
applies to all your repositories. This repo's `CLAUDE.md` already states the rule, so
Claude Code should pick it up automatically — the CI job is the backstop for when it
does not, or for other tooling.

### If you have already pushed one

Amend the commit message and force-push your branch. Trailers already merged into `main`
are left alone: rewriting shared history would change every commit id and break every
other clone, which costs more than it fixes.
