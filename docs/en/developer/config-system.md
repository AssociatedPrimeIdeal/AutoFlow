# Configuration system

AutoFlow keeps defaults in per-module JSON files. The complete key/type/default/effect/owner tables are maintained in [Configuration parameters](../user/parameters.md); nested groups and learned-backend options are in [Structured parameters](../user/parameter-schemas.md).

## Precedence

| Entry point | Resolution order |
| --- | --- |
| CLI | Built-ins in `autoflow/config.py` → selected `configs/*.json` → explicit flags |
| GUI | Built-ins → selected JSON bundle → interactive edits |
| Python `AutoFlowConfig()` | Dataclass class defaults → explicit constructor fields |
| Python `AutoFlowConfig.from_config_dir()` | Built-ins → selected JSON bundle → explicit keyword overrides |

`build_workspace(config)` loads the bundle selected by `config.config_dir`, then applies mapped public API fields. Numerical fields without a direct dataclass member, such as WSS smoothing or PG erosion, come from the module bundle. Do not assume the dataclass defaults equal the shipped JSON: for example direct API WSS `clim` is `[0,10]`, whereas the shipped bundle uses `[0,5]`.

## Ownership

`autoflow/config.py` defines defaults, deep merging, legacy compatibility and bundle-to-API/workspace conversion. `autoflow/api.py` defines public runtime overrides. `autoflow/core/models.py` owns workspace parameter objects. CLI parsing lives in `autoflow/cli.py`; GUI configuration and runtime edits live in `autoflow/ui/app.py`.

Metric-specific JSON density/viscosity keys can override shared `fluid.json` values. Otherwise WSS and pressure use shared viscosity, and pressure/TKE use shared density. `derived.json` is a compatibility placeholder; use the named metric modules for new settings.

GUI metric layers share the live `colorbar.json` layout. Metric `render.bar_cfg` controls offline video defaults; GUI runtime colourbar edits can also be copied into video settings for that session.

## Updating parameters

1. Add or change the built-in and shipped JSON default.
2. Update the workspace model and config mapping, plus a dataclass field / CLI flag only if it is public there.
3. Update `docs/en/developer/parameter-descriptions.json` with meaning, units and ownership.
4. Update the affected rows in the configuration, structured-parameter, CLI and API references. Keep type, default, configuration location, effect and code owner synchronized with the source.
5. Update the affected feature's behaviour and limitation text.
6. Compare the edited references with the live merged bundle, dataclass fields and argparse declarations, then run `mkdocs build --strict`.

Parameter references are maintained alongside code changes. Review every new or changed control against its declaration and effective shipped default; documentation-site builds check links and rendering, not numerical parameter completeness.
