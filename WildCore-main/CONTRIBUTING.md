# Contributing to WildCore

We are excited about your interest in contributing! Here are some guidelines to help you get started.

## Reporting Bugs

* Use the GitHub **Issues** section to report bugs.
* Be as detailed as possible. Include your Python version, dependencies, and the steps to reproduce the bug.

## Suggesting Enhancements

* Use the **Issues** section to suggest new features. Explain the use case and why the enhancement would be useful.

## Pull Request Process

1. **Fork** the repository.
2. Create a new branch for your feature (`git checkout -b feature/new-feature`).
3. Commit your changes (`git commit -m 'Adds new-feature'`).
4. Push to the branch (`git push origin feature/new-feature`).
5. Open a **Pull Request**.

## Code Style

* Follow PEP 8 guidelines.
* Include docstrings for all functions, classes, and methods.
* Write tests for new features.

## Development Setup

1. Clone your fork of the repository
2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. Install development dependencies:
```bash
pip install .[dev]
```
4. Run tests before submitting a PR:
```bash
pytest
```

## Code Review

All submissions require review. We use GitHub pull requests for this purpose.

## Community

* Be respectful and inclusive.
* Follow our [Code of Conduct](CODE_OF_CONDUCT.md).
* Help others who have questions.

Thank you for contributing to WildCore!
