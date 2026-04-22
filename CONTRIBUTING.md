# Contributing to LiteRAG

Thank you for contributing. This guide explains how to propose changes, set up your environment, and submit production-ready pull requests.

## Ways to Contribute

- Report bugs using the issue templates.
- Propose features or design improvements.
- Improve docs, examples, and developer experience.
- Contribute code for performance, reliability, and test coverage.

## Before You Start

- Search existing issues and pull requests to avoid duplicates.
- Open an issue for large or breaking changes before implementation.
- Keep pull requests focused and small enough to review quickly.

## Development Setup

1. Fork and clone.
1. Create and activate a virtual environment.
1. Install runtime and development dependencies.

```bash
git clone <your-fork-url>
cd LiteRAG
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

1. Configure local environment:

```bash
cp .env.example .env
```

Set required environment variables (for example GEMINI_API_KEY and LiteRAG_DATA_DIR) before running examples.

1. Place GraphRAG parquet outputs under your selected data directory.

## Local Validation

Run these before opening a pull request:

```bash
pytest -q
```

If you change core retrieval behavior, include a short note in the pull request with:

- Dataset size and characteristics.
- Config values used.
- Before/after behavior or latency.

## Code Guidelines

- Follow existing module boundaries and naming patterns.
- Keep functions small and cohesive.
- Add docstrings for public classes/functions and non-obvious logic.
- Avoid introducing breaking changes without prior discussion.
- Update documentation with every behavior or API change.

## Commit and Branch Conventions

- Branch naming examples:
  - feature/anchor-scoring-tuning
  - fix/context-token-budget
  - docs/api-reference-update
- Conventional Commits are recommended:
  - feat: add configurable safety net table name
  - fix: guard null community report fields
  - docs: clarify async query usage

## Pull Request Checklist

Before opening your pull request, verify:

- The change has a clear problem statement.
- Tests pass locally.
- Documentation is updated where needed.
- Sensitive data is not committed.
- The pull request template is fully completed.

## Reporting Security Issues

Do not file public issues for security vulnerabilities. See [SECURITY.md](SECURITY.md).

## Code of Conduct

By participating, you agree to follow [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
