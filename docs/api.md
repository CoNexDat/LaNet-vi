# API reference

The package is used through `Network`: load a graph, `decompose()`, then `visualize()`
(the layout is computed on the way, or explicitly with `compute_layout()`). Every setting
is a field of `LaNetConfig`, the same model the CLI fills from its flags. The
[usage guide](usage.md#python-api) walks through a complete example.

`Network`, `LaNetConfig` and its sub-models, `DecompositionType`, `ColorScheme`,
`BackgroundColor`, `load_config_from_yaml` and `save_config_to_yaml` are re-exported
from the top-level `lanet_vi` package; everything else is imported from its module.

## Network

::: lanet_vi.core.network

## Configuration

::: lanet_vi.models.config

## Results

::: lanet_vi.models.graph

## Decomposition

::: lanet_vi.decomposition.kcores

::: lanet_vi.decomposition.kdenses

::: lanet_vi.decomposition.dcores

## Layout

::: lanet_vi.visualization.lanet_layout

::: lanet_vi.visualization.layout

## Rendering

::: lanet_vi.visualization.matplotlib_renderer

::: lanet_vi.visualization.colors

## Input and output

::: lanet_vi.io.readers

::: lanet_vi.io.writers

::: lanet_vi.io.config_loader

## Generators

::: lanet_vi.generators

## Metrics

::: lanet_vi.metrics

## Community detection

::: lanet_vi.community
