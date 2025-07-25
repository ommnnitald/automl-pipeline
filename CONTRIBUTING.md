# Contributing to AutoMLPipeline 🤝

We love your input! We want to make contributing to AutoMLPipeline as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## 🚀 Quick Start for Contributors

### 1. Fork & Clone
```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/automl-pipeline.git
cd automl-pipeline
```

### 2. Set Up Development Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### 3. Create Feature Branch
```bash
git checkout -b feature/amazing-new-feature
```

### 4. Make Changes & Test
```bash
# Run tests
pytest

# Run linting
black src/ tests/
flake8 src/ tests/
mypy src/

# Run full test suite
tox
```

### 5. Submit Pull Request
```bash
git add .
git commit -m "Add amazing new feature"
git push origin feature/amazing-new-feature
```

Then create a Pull Request on GitHub!

---

## 📋 Development Guidelines

### Code Style
- **Python Style**: Follow [PEP 8](https://pep8.org/)
- **Formatting**: Use [Black](https://black.readthedocs.io/) for code formatting
- **Linting**: Use [flake8](https://flake8.pycqa.org/) for linting
- **Type Hints**: Use [mypy](http://mypy-lang.org/) for type checking
- **Docstrings**: Follow [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)

### Testing
- **Framework**: Use [pytest](https://pytest.org/) for testing
- **Coverage**: Maintain >90% test coverage
- **Types**: Write unit tests, integration tests, and end-to-end tests
- **Location**: Tests go in `tests/` directory

### Documentation
- **API Docs**: Use docstrings for all public functions/classes
- **User Docs**: Update relevant documentation in `docs/`
- **Examples**: Add examples for new features
- **Changelog**: Update `CHANGELOG.md` for user-facing changes

---

## 🐛 Bug Reports

Great Bug Reports tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

**Use our bug report template** when creating issues.

---

## 💡 Feature Requests

We love feature requests! Before submitting:

1. **Check existing issues** to avoid duplicates
2. **Provide clear use case** - why is this feature needed?
3. **Consider scope** - is this a core feature or plugin?
4. **Think about API** - how should users interact with this feature?

**Use our feature request template** when creating issues.

---

## 🔄 Pull Request Process

### Before Submitting
- [ ] Tests pass locally (`pytest`)
- [ ] Code is formatted (`black src/ tests/`)
- [ ] Linting passes (`flake8 src/ tests/`)
- [ ] Type checking passes (`mypy src/`)
- [ ] Documentation is updated
- [ ] Changelog is updated (if user-facing)

### PR Requirements
- **Clear Description**: Explain what and why
- **Link Issues**: Reference related issues
- **Small Scope**: Keep PRs focused and reviewable
- **Tests**: Include tests for new functionality
- **Documentation**: Update docs for new features

### Review Process
1. **Automated Checks**: CI must pass
2. **Code Review**: At least one maintainer review
3. **Testing**: Verify functionality works as expected
4. **Documentation**: Ensure docs are clear and complete

---

## 🏗️ Development Setup

### Prerequisites
- Python 3.8+
- Git
- Virtual environment tool (venv, conda, etc.)

### Full Development Setup
```bash
# Clone repository
git clone https://github.com/automl-pipeline/automl-pipeline.git
cd automl-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e ".[dev,docs,test]"

# Install pre-commit hooks
pre-commit install

# Verify installation
python -c "import automl_pipeline; print('Success!')"
```

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=automl_pipeline --cov-report=html

# Run specific test file
pytest tests/unit/test_core/test_pipeline.py

# Run tests with specific marker
pytest -m "not slow"
```

### Building Documentation
```bash
cd docs/
make html
# Open docs/build/html/index.html
```

---

## 📁 Project Structure

```
automl-pipeline/
├── src/automl_pipeline/     # Main package
│   ├── core/               # Core functionality
│   ├── stages/             # Pipeline stages
│   ├── utils/              # Utilities
│   └── models/             # Model implementations
├── tests/                  # Test suite
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── fixtures/          # Test data
├── docs/                  # Documentation
├── examples/              # Example scripts
├── benchmarks/            # Performance tests
└── .github/               # GitHub workflows
```

---

## 🎯 Contribution Areas

### 🐛 Bug Fixes
- Fix reported issues
- Improve error handling
- Add edge case handling

### ✨ New Features
- New ML algorithms
- Additional data preprocessing
- Enhanced reporting
- Performance optimizations

### 📖 Documentation
- API documentation
- Tutorials and examples
- User guides
- Developer documentation

### 🧪 Testing
- Unit test coverage
- Integration tests
- Performance benchmarks
- Edge case testing

### 🎨 Examples
- Real-world use cases
- Tutorial notebooks
- Best practices guides
- Industry-specific examples

---

## 🏷️ Issue Labels

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Improvements or additions to documentation
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention is needed
- `question`: Further information is requested
- `wontfix`: This will not be worked on

---

## 📞 Getting Help

- **GitHub Discussions**: For questions and general discussion
- **Discord**: Real-time chat with the community
- **Stack Overflow**: Tag questions with `automl-pipeline`
- **Email**: Contact maintainers directly

---

## 🙏 Recognition

Contributors will be:
- Listed in `AUTHORS.md`
- Mentioned in release notes
- Invited to join the contributor team
- Given credit in documentation

---

## 📜 Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

---

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to AutoMLPipeline! 🎉**

---

## 🏗️ Development Setup

### Prerequisites
- Python 3.8+
- Git
- Virtual environment tool (venv, conda, etc.)

### Full Development Setup
```bash
# Clone repository
git clone https://github.com/automl-pipeline/automl-pipeline.git
cd automl-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e ".[dev,docs,test]"

# Install pre-commit hooks
pre-commit install

# Verify installation
python -c "import automl_pipeline; print('Success!')"
```

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=automl_pipeline --cov-report=html

# Run specific test file
pytest tests/unit/test_core/test_pipeline.py

# Run tests with specific marker
pytest -m "not slow"
```

### Building Documentation
```bash
cd docs/
make html
# Open docs/build/html/index.html
```

---

## 📁 Project Structure

```
automl-pipeline/
├── src/automl_pipeline/     # Main package
│   ├── core/               # Core functionality
│   ├── stages/             # Pipeline stages
│   ├── utils/              # Utilities
│   └── models/             # Model implementations
├── tests/                  # Test suite
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── fixtures/          # Test data
├── docs/                  # Documentation
├── examples/              # Example scripts
├── benchmarks/            # Performance tests
└── .github/               # GitHub workflows
```

---

## 🎯 Contribution Areas

### 🐛 Bug Fixes
- Fix reported issues
- Improve error handling
- Add edge case handling

### ✨ New Features
- New ML algorithms
- Additional data preprocessing
- Enhanced reporting
- Performance optimizations

### 📖 Documentation
- API documentation
- Tutorials and examples
- User guides
- Developer documentation

### 🧪 Testing
- Unit test coverage
- Integration tests
- Performance benchmarks
- Edge case testing

### 🎨 Examples
- Real-world use cases
- Tutorial notebooks
- Best practices guides
- Industry-specific examples

---

## 🏷️ Issue Labels

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Improvements or additions to documentation
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention is needed
- `question`: Further information is requested
- `wontfix`: This will not be worked on

---

## 📞 Getting Help

- **GitHub Discussions**: For questions and general discussion
- **Discord**: Real-time chat with the community
- **Stack Overflow**: Tag questions with `automl-pipeline`
- **Email**: Contact maintainers directly

---

## 🙏 Recognition

Contributors will be:
- Listed in `AUTHORS.md`
- Mentioned in release notes
- Invited to join the contributor team
- Given credit in documentation

---

## 📜 Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

---

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to AutoMLPipeline! 🎉**