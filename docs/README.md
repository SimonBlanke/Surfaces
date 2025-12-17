# Surfaces Documentation

This directory contains the Sphinx documentation for the Surfaces library.

## Building the Documentation

### Prerequisites

Install the documentation dependencies:

```bash
pip install -r requirements.txt
```

### Building HTML Documentation

From the `docs/source` directory:

```bash
make html
```

The built documentation will be in `docs/build/html/`.

### Live Preview

For development with auto-rebuild:

```bash
make livehtml
```

This starts a server at `http://127.0.0.1:8000` that auto-rebuilds on changes.

### Cleaning Build Artifacts

```bash
make clean
```

## Structure

```
docs/
├── build/              # Built documentation output
├── requirements.txt    # Documentation dependencies
├── README.md          # This file
└── source/            # Documentation source files
    ├── conf.py        # Sphinx configuration
    ├── index.rst      # Main page
    ├── _static/       # Static files (CSS, images)
    ├── _templates/    # Custom templates
    ├── api_reference/ # API documentation
    ├── examples/      # Code examples
    └── user_guide/    # User guide pages
```

## Theme

The documentation uses the pydata-sphinx-theme with a custom dark pastel red color scheme. Custom CSS is in `source/_static/css/custom.css`.
