# Getting Started

New to PICASO? Three steps take you from install to your first spectrum.

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} 1. Install
:link: installation
:link-type: doc

Install with pip, conda or from source, plus the optional driver and app add-ons.
:::

:::{grid-item-card} 2. Get data with the Quickstart
:link: notebooks/Quickstart
:link-type: doc

Set `picaso_refdata`, download the reference data and opacities with `get_data()`, and check your environment. Students and workshops can grab the lighter *picaso-lite* set here too.
:::

:::{grid-item-card} 3. Compute your first spectrum
:link: notebooks/A_basics/1_GetStarted
:link-type: doc

Learn PICASO's basic inputs and outputs, then continue through the rest of the tutorials.
:::

::::

:::{tip}
Prefer to set things up by hand, or need optional data such as stellar grids, correlated-k tables or Sonora models? See {ref}`reference_data` on the Installation page, which also explains how to set environment variables permanently.
:::

The tutorials are Jupytext `.py` files that you can run as notebooks; {doc}`notebook_workflow` explains how to download and open them.

```{toctree}
:hidden:
:maxdepth: 2

installation
notebooks/Quickstart.py
notebook_workflow
```
