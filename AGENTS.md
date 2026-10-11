# AGENTS.md

## Public Repo Safety

This repo is intended to be safe to publish publicly.

Do not commit:

- credentials
- bundled transcript outputs
- local model files
- downloaded audio
- private config

Models and transcription artifacts should stay outside the repo unless they are small, synthetic fixtures.

## Development workflow

GitHub Issues are canonical for repo-bound bugs, features, refactors and release tasks. Local Markdown supports scratch thinking. Read [issue-tracker.md](docs/agents/issue-tracker.md), [triage-labels.md](docs/agents/triage-labels.md) and [domain.md](docs/agents/domain.md) for repository context before publishing or labeling work.

- Define the requested outcome, acceptance criteria and verification needed for the change; report done, blocked, parked or split accurately.
- Use skills available in the current session when they help the task. Follow the user's skill and delegation preferences; do not assume a workflow package is installed.
- Keep small fixes within the requested scope. Planning, worktrees, delegation and deployment gates should match the work and its authority.
- For code changes, run `python -m unittest discover -s tests` and check relevant CLI help. Use synthetic fixtures; do not transcribe private audio or mutate the live import ledger for verification.
