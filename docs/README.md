# pyFANTOM Documentation

This directory contains the Sphinx documentation for pyFANTOM.

## Building the Documentation

### Prerequisites

Install the documentation dependencies:

```bash
pip install -r docs/requirements.txt
```

Or install them manually:

```bash
pip install sphinx sphinx-rtd-theme myst-parser
```

### Building HTML Documentation

From the `docs` directory:

```bash
cd docs
make html
```

The generated HTML documentation will be in `docs/build/html/`. Open `docs/build/html/index.html` in your browser to view it.

### Building Other Formats

- **PDF**: `make latexpdf` (requires LaTeX)
- **Link Check**: `make linkcheck` (checks all links)
- **Clean**: `make clean` (removes build files)

## Documentation Structure

- `source/index.rst` - Main documentation index
- `source/getting_started.rst` - Getting started guide (placeholder)
- `source/examples.rst` - Examples and tutorials (placeholder)
- `source/contributing.rst` - Contributing guidelines (placeholder)
- `source/api/` - Automatically generated API documentation

## Adding Content

1. **Getting Started**: Edit `source/getting_started.rst` to add installation instructions and basic usage
2. **Examples**: Edit `source/examples.rst` to add links to your Jupyter notebooks or code examples
3. **API Documentation**: The API docs are automatically generated from docstrings. Add docstrings to your code following NumPy or Google style.

## Configuration

The main configuration file is `source/conf.py`. Key settings:

- **Theme**: Read the Docs theme (sphinx_rtd_theme)
- **Extensions**: autodoc, autosummary, napoleon (for NumPy/Google docstrings), viewcode, intersphinx
- **Mock imports**: Some optional dependencies (like cupy) are mocked to allow building docs without them

