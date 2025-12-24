# Surfaces Documentation

Sphinx-based documentation using automation to scale with the growing number of test functions.

## Design Principle

**Automation over manual updates.** Adding a new test function should not require editing documentation files. The docs regenerate from source code.

## Architecture

```
docs/
├── _generators/           # Python scripts that generate RST content
│   ├── generate_catalogs.py   # Function tables by category
│   ├── generate_plots.py      # Surface/contour plots for 2D functions
│   ├── generate_diagrams.py   # Module structure diagrams
│   └── generate_all.py        # Master script
├── source/
│   ├── _generated/        # Auto-generated content (gitignored)
│   │   ├── catalogs/      # Function tables as RST
│   │   ├── plots/         # PNG visualizations
│   │   └── diagrams/      # Architecture diagrams
│   ├── _templates/        # Custom autosummary templates
│   ├── api_reference/     # Uses autosummary (no manual per-function RST)
│   └── user_guide/        # Narrative docs with included generated tables
└── requirements.txt
```

## What Gets Generated

| Content | Generator | Trigger |
|---------|-----------|---------|
| Function catalog tables | `generate_catalogs.py` | Functions in `__all__` |
| Surface/contour plots | `generate_plots.py` | 2D functions with `n_dim=2` |
| Module hierarchy diagram | `generate_diagrams.py` | Package structure |
| API reference pages | Sphinx autosummary | Docstrings |
| Function counts | `conf.py` | Runtime introspection |

## Adding a New Test Function

1. Create the class in the appropriate module
2. Add it to `__all__` in the module's `__init__.py`
3. Run `make docs`

The generators handle everything else: API page, catalog entry, plots (if 2D), count updates.

## Building

```bash
# Install dependencies
pip install -r docs/requirements.txt

# Full build (generate + sphinx)
make docs

# Just regenerate assets
make docs-generate

# Just sphinx (faster, uses cached generated content)
make docs-quick

# Clean all generated content
make docs-clean
```

## Manual vs Generated Content

**Manual (narrative, curated):**
- `index.rst` - Landing page structure
- `user_guide/*.rst` - Tutorials and explanations (include generated tables via RST directives)
- `examples/*.rst` - Code examples
- `api_reference/base.rst` - Base class documentation

**Generated (automated):**
- `_generated/catalogs/*.rst` - Function tables
- `_generated/plots/*.png` - Visualizations
- `_generated/diagrams/*.rst` - Architecture diagrams
- `_templates/autosummary/` stubs - Individual API pages

## Scalability

The current structure supports 100+ functions without documentation overhead:

- **No per-function RST files**: Autosummary generates API pages from docstrings
- **Declarative registration**: Add to `__all__`, docs update automatically
- **Cached plot generation**: Only regenerates when function source changes
- **Grouped catalogs**: Functions organized by category, avoiding endless lists
- **Dynamic counts**: All statistics derived from source at build time

## File Counts Target

| Type | Current | Target |
|------|---------|--------|
| Manual RST files | ~15 | <15 |
| Generated RST files | ~20+ | Scales with functions |
| Per-function effort | 0 min | 0 min |

## Related Documentation

Detailed specifications in `/home/me/github-workspace/999-private/project-planning/Surfaces/docs/`:
- `001-documentation-strategy.md` - Full strategy and rationale
- `002-generator-specifications.md` - Generator implementation details
- `003-implementation-roadmap.md` - Phased implementation plan
