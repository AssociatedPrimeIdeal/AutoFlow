# Architecture

## Purpose
This page explains where the major runtime responsibilities live so maintainers can find the right layer quickly.

## Main Layers

| Layer | Directory | Responsibility |
| --- | --- | --- |
| public entry points | `autoflow/` | CLI, GUI launcher, public Python API |
| pipeline orchestration | `autoflow/core` | workspace state and step orchestration |
| algorithms | `autoflow/algorithms` | loading, preprocessing, graph, planes, metrics, segmentation, streamlines |
| GUI | `autoflow/ui` | Qt window, docks, viewers, dialogs, editors |
| rendering | `autoflow/rendering` | offline video export |
| tests | `tests` | behavior coverage |
| configuration | `configs`, `autoflow/config.py` | per-module JSON defaults and loading |

## Primary Execution Paths

### CLI batch path
1. `autoflow/cli.py` parses flags
2. `autoflow/api.py` builds `AutoFlowConfig`
3. `autoflow/processing.py:process_single()` runs the batch order
4. `autoflow/core/pipeline.py` executes concrete steps

### GUI path
1. `autoflow/ui/launcher.py` starts the app
2. `autoflow/ui/app.py` manages the main window and UI state
3. `autoflow/core/pipeline.py` runs steps against the workspace
4. `autoflow/ui/viewer.py` and `autoflow/ui/ortho_viewer.py` render 3D and 2D state

### Python API path
1. `autoflow/api.py` exposes `AutoFlowConfig`, `run_case()`, `run_batch()`, and `build_workspace()`
2. config defaults come from `autoflow/config.py`
3. execution still flows through `autoflow/processing.py` and `autoflow/core/pipeline.py`

## Core Data Flow

1. loader returns `LoadedCase`
2. workspace stores normalized source arrays and metadata
3. segmentation availability controls downstream step eligibility
4. skeleton feeds graph and paths
5. graph and paths feed planes
6. planes feed metrics
7. segmentation and flow feed derived metrics
8. rendering consumes planes, metrics, and derived arrays

## Design Constraints To Keep
- `LoadedCase` normalization is the contract between loaders and the rest of the system
- TKE stays optional
- segmentation is an interface with multiple valid sources
- GUI and CLI share pipeline logic as much as possible
