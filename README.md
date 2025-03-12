# DSPy Prompt Generator

A Flask web application for generating and optimizing prompts using DSPy.

## Project Structure

```
promptgen/
├── app/                    # Main application package
│   ├── __init__.py         # Application factory
│   ├── config.py           # Configuration settings
│   ├── models/             # Data models
│   │   ├── __init__.py
│   │   └── signature.py    # Signature definition model
│   ├── routes/             # Flask route definitions
│   │   ├── __init__.py
│   │   ├── api.py          # API routes
│   │   ├── main.py         # Main routes
│   │   ├── programs.py     # Program management routes
│   │   ├── samples.py      # Sample management routes
│   │   └── signatures.py   # Signature management routes
│   ├── services/           # Business logic services
│   │   ├── __init__.py
│   │   ├── evaluation.py   # Evaluator service
│   │   ├── optimization.py # Optimizer service
│   │   ├── samples.py      # Sample management service
│   │   └── state.py        # Application state service
│   ├── static/             # Static assets
│   │   ├── css/
│   │   └── js/
│   ├── templates/          # Jinja2 templates
│   └── utils/              # Utility functions
│       ├── __init__.py
│       ├── filters.py      # Template filters
│       └── metrics.py      # Evaluation metrics
├── programs/               # Storage for DSPy programs
├── samples/                # Storage for samples
├── signatures/             # Storage for signatures
├── CLAUDE.md               # Instructions for Claude
├── pyproject.toml          # Project metadata and dependencies
├── README.md               # Project documentation
└── wsgi.py                 # WSGI entry point
```

## Features

- Create and manage DSPy signatures for different tasks
- Generate programs for different signature types
- Collect training samples for each signature
- Optimize programs using DSPy optimizers
- Evaluate program performance
- Import and export samples

## Setup

1. Install dependencies:
```bash
uv sync
```

2. Run the cleanup script to remove old files (if you're upgrading from an older version):
```bash
./cleanup.sh
```

3. Run the URL reference update script to fix template links:
```bash
python3 update_url_references.py
```

4. Run the development server:
```bash
flask --app wsgi run --debug
```

5. Access the application at http://localhost:5000

### Note for Upgraders

The codebase has been restructured with routes organized into blueprints:
- `main`: Main dashboard and redirects
- `signatures`: Signature management
- `samples`: Sample management
- `programs`: Program management
- `api`: JSON API endpoints

URL references in templates and code should be updated to reflect these changes, but backward compatibility redirects have been added for common routes.

## Creating a New Signature

1. Navigate to the Signatures page
2. Click "Add New Signature"
3. Fill in the signature details:
   - Name (class name)
   - Description
   - Input Fields (comma-separated)
   - Output Fields (comma-separated)
   - Full DSPy signature class definition

## Working with Samples

Samples can be created, imported, and exported for each signature:

- Add new samples manually through the UI
- Import samples from JSON files
- Export samples to JSON files for sharing or backup

## Optimization

1. Select a signature and ensure it has samples
2. Click "Start Optimization" to begin DSPy optimization
3. Once complete, the new program will be automatically selected

## Dependencies

- Flask: Web framework
- DSPy: Deep learning framework for prompting
- Bootstrap 5: Frontend framework (via CDN)
- jQuery: JavaScript library (via CDN)
