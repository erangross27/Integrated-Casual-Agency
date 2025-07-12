# Requirements Directory

This directory contains all dependency specifications for the ICA Framework.

## Files

### `requirements.txt`
Core dependencies required for basic ICA Framework functionality:
- Essential packages: numpy, networkx, torch, scikit-learn, scipy
- Framework utilities: pydantic, click, loguru
- Database support: neo4j
- Development tools: pytest, black, isort

### `requirements-dev.txt`
Development dependencies including all core requirements plus:
- Development tools: jupyter, notebook, mypy
- Enhanced visualization: seaborn, plotly, pandas, tqdm

### `REQUIREMENTS_CONSOLIDATION.md`
Documentation of the requirements consolidation process, including:
- What was changed from the original multi-file structure
- Package analysis and decisions
- Installation options and benefits

## Installation Options

```bash
# Core functionality only
pip install -r requirements/requirements.txt

# Development setup (includes core + dev tools)
pip install -r requirements/requirements-dev.txt

# Feature-specific installation via setup.py extras
pip install -e .[viz]        # Visualization packages
pip install -e .[ml]         # Machine learning packages  
pip install -e .[causal]     # Causal inference packages
pip install -e .[monitoring] # Progress tracking packages
pip install -e .[all]        # All optional packages
```

## Design Philosophy

The requirements are organized to:
- **Minimize installation time** - Core requirements install quickly
- **Reduce complexity** - Two main files instead of multiple scattered files
- **Enable flexibility** - Optional packages available via extras
- **Support development** - Clear separation of dev vs runtime dependencies
- **Maintain compatibility** - Handles multiple package formats and versions
