# 🤝 Contributing Guidelines

Thank you for considering a contribution to this project! To keep things clean and collaborative, please follow the steps below.

---

## ⚙️ Setup

This project uses [Poetry](https://python-poetry.org/) for dependency management.

```bash
# Install dependencies (including dev tools)
poetry install

# Activate the virtual environment
poetry shell
```

---

## 🔀 Workflow: How to Contribute

1. **Checkout from `dev`**
   - Always create new branches from the `dev` branch:

     ```bash
     git checkout dev
     git pull
     git checkout -b your-feature-branch
     ```

2. **Make your changes**
   - Write clean, tested, and documented code.
   - Follow the style guide below.

3. **Run tests locally before pushing**
   - Make sure all tests pass and coverage meets the 90% threshold:

     ```bash
     poetry run pytest --cov=src --cov-report=term-missing --cov-fail-under=90
     ```

   - If coverage is below 90%, add tests before opening a PR.

4. **Commit your changes**
   - Use clear commit messages (see below).

5. **Push your branch**

   ```bash
   git push origin your-feature-branch
   ```

6. **Create a Pull Request**
   - Always target the `dev` branch.
   - Provide a clear description of your changes.
   - Link any related issues.
   - GitHub will automatically require:
     - ✅ A pull request
     - ✅ At least 1 approval
     - ✅ CI checks passing (tests + 90% coverage)

---

## 🧪 Testing

We use [pytest](https://pytest.org/) with [pytest-cov](https://pytest-cov.readthedocs.io/) for test coverage.

### Running tests

```bash
# Run all tests
poetry run pytest

# Run with coverage report
poetry run pytest --cov=src --cov-report=term-missing

# Run a specific file or test
poetry run pytest tests/test_mymodule.py
poetry run pytest tests/test_mymodule.py::test_my_function
```

### Coverage requirements

- **Minimum: 90%** — PRs will be blocked by CI if coverage drops below this threshold.
- Place all tests in the `tests/` directory.
- Test files must be named `test_*.py` or `*_test.py`.
- Aim to test edge cases, not just the happy path.

### Excluding files from coverage

Some files don't need coverage (config, migrations, etc.). Add them to `pyproject.toml`:

```toml
[tool.coverage.run]
omit = [
    "src/config/*",
    "src/*/__init__.py",
]
```

---

## 📦 Commit Message Guidelines

Please use clear and consistent commit titles. Good commit messages help others understand your changes quickly.

### ✅ Good Examples

- `Fix login redirect issue on Firefox`
- `Add unit tests for payment service`
- `Refactor user model for better readability`

### ❌ Bad Examples

- `Update`
- `Bugfix`
- `Misc changes`
- `temp`
- `Fix some stuff`

### 🛠️ Optional Format (Recommended)

```
<type>: <short description>
```

#### Common types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only changes
- `style`: Formatting, missing semi colons, etc.
- `refactor`: Code change that neither fixes a bug nor adds a feature
- `test`: Adding or fixing tests
- `chore`: Other changes (e.g., build tools, CI)

##### Examples

- `feat: add user profile settings page`
- `fix: correct typo in error message`
- `chore: update dependencies`

---

## 🔒 Branch Protection Rules

The following rules are active on `main` and `dev`:

- ✅ Pull request required before merging
- ✅ At least 1 approval required
- ✅ Stale approvals dismissed when new commits are pushed
- ✅ Conversation resolution required before merging
- ✅ Force pushes blocked
- ✅ CI status checks must pass (tests + 90% coverage)

Do not push directly to `main` or `dev`.
