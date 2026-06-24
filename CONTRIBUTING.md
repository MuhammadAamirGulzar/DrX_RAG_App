# Contributing

## Development Principles

- Keep changes scoped and reviewable.
- Prefer explicit behavior over hidden side effects.
- Preserve retrieval traceability in all user-facing responses.

## Local Setup

1. Create an environment using `environment.yml` or `requirements.txt`.
2. Download NLTK tokenization data (`punkt`).
3. Run the app with `streamlit run app.py`.

## Pull Request Guidelines

- Open focused pull requests with a clear problem statement.
- Include validation notes for ingestion, retrieval, and generation paths.
- Update documentation when behavior or configuration changes.

## Commit Message Format

Use Conventional Commits:

- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation updates
- `refactor:` for internal structure changes
- `chore:` for maintenance tasks
- `test:` for tests and validation improvements

Examples:

- `feat: add semantic grouping for csv chunking`
- `fix: guard summary parsing against unsafe eval`
- `docs: clarify local model cache behavior`

## Code Style

- Keep functions cohesive and single-purpose where practical.
- Use explicit metadata keys for chunk provenance.
- Avoid introducing hardcoded machine-specific paths.
