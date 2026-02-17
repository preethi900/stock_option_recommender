# Buildkite Pipeline Setup 🚀

This directory contains a simple Buildkite CI pipeline configuration for the Stock Option Recommender project that checks code quality on every commit.

## Files

- **pipeline.yml** - Buildkite pipeline configuration (pylint checks only)
- **buildkite-editor.html** - Visual YAML editor for creating/editing pipeline steps

## Pipeline Overview

The pipeline is intentionally simple and focused:

**� Pylint Code Check** - Runs code quality checks on all Python files
- Installs project dependencies
- Runs pylint on all `.py` files in the repository
- Uses Python 3.10 in a Docker container
- Triggers automatically on every push

## Getting Started

### 1. Connect to Buildkite

1. Create a new pipeline in your Buildkite dashboard
2. Point it to this repository
3. Set the pipeline file path to `.buildkite/pipeline.yml`

### 2. That's It! 🎉

The pipeline will automatically run pylint checks on every commit. No additional configuration needed.

## Running Pylint Locally

Before pushing, you can run pylint locally to catch issues:

```bash
# Activate your virtual environment
source venvironment/bin/activate

# Install pylint if not already installed
pip install pylint

# Run pylint on all Python files (same as Buildkite)
pylint $(git ls-files '*.py')
```

## Customizing Pylint Rules

To customize pylint behavior, you can:

### Option 1: Add command-line flags in `pipeline.yml`

```yaml
pylint $(git ls-files '*.py') --max-line-length=100 --disable=C0114
```

### Option 2: Create a `.pylintrc` file

Create a `.pylintrc` file in your project root:

```ini
[MASTER]
disable=C0114,C0115,C0116

[FORMAT]
max-line-length=100

[MESSAGES CONTROL]
# Add any specific rules you want to disable
```

## Using the Visual Editor

Want to add more steps (tests, deployment, etc.)? Use the visual editor:

```bash
# Open the editor in your browser
open ../buildkite-editor.html
```

Features:
- 🎨 Visual step builder with drag-and-drop
- 📋 Real-time YAML preview
- ⬇️ Download generated YAML
- 🐳 Docker plugin integration

## Troubleshooting

### Pylint Fails on Buildkite but Passes Locally

Make sure you're testing with the same Python version:
```bash
# The pipeline uses Python 3.10
python3.10 -m pip install pylint
python3.10 -m pylint $(git ls-files '*.py')
```

### Want to Skip Pylint for a Specific Commit?

Add `[skip ci]` to your commit message:
```bash
git commit -m "Update documentation [skip ci]"
```

### Fixing Common Pylint Errors

**Line too long:**
```bash
# Break long lines or add max-line-length to pipeline.yml
pylint $(git ls-files '*.py') --max-line-length=100
```

**Missing docstrings:**
```bash
# Disable if not needed
pylint $(git ls-files '*.py') --disable=C0114,C0115,C0116
```

## Expanding the Pipeline

Need more than just pylint? Here are some ideas:

**Add Testing:**
```yaml
  - label: ":test_tube: Run Tests"
    key: "test"
    command: |
      pip install -r requirements.txt
      pip install pytest pytest-cov
      pytest --cov
```

**Add Type Checking:**
```yaml
  - label: ":mag: Type Check with mypy"
    key: "mypy"
    command: |
      pip install -r requirements.txt
      pip install mypy
      mypy .
```

**Add Security Scanning:**
```yaml
  - label: ":lock: Security Scan"
    key: "security"
    command: |
      pip install bandit
      bandit -r .
```

Use the visual editor (`buildkite-editor.html`) to build these steps visually!

## Resources

- [Buildkite Documentation](https://buildkite.com/docs)
- [Pylint Documentation](https://pylint.readthedocs.io/)
- [Docker Plugin Docs](https://github.com/buildkite-plugins/docker-buildkite-plugin)

## Support

For issues:
1. Check Buildkite build logs
2. Run pylint locally to reproduce
3. Open an issue on GitHub

Happy linting! 🎯
