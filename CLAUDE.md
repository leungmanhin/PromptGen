# Project Guide for PromptGen

## Commands
- **Setup:** `uv sync` - Install dependencies
- **Run App:** `flask --app app run --debug` - Start development server
- **Optimize:** `flask optimize` - Run optimization from CLI
- **Run Tests:** `pytest` - Run test suite
- **Run Single Test:** `pytest path/to/test.py::test_function`
- **Lint:** `ruff check .` - Check code style

## Code Style Guidelines
- **Imports:** System → Third-party → Local, alphabetically sorted
- **Types:** Use type hints for all functions and variables
- **Naming:** Snake_case for variables/functions, PascalCase for classes
- **Documentation:** Docstrings for all public functions and classes
- **Error Handling:** Use try/except with detailed error messages
- **Architecture:** Maintain separation of concerns (samples, evaluation, optimization)
- **Function Parameters:** Named parameters with type annotations
- **Code Structure:** Modular design with clear component responsibilities

## Important
- Always add tests for new features

This is a Flask-based application for prompt generation and optimization.
