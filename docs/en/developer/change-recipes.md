# Change Recipes

## Add A New CLI Option
1. edit `autoflow/cli.py`
2. map the flag into `AutoFlowConfig` in `autoflow/api.py` when it should be public
3. update `autoflow/config.py` and the right `configs/<module>.json` file when it needs a config default
4. update `build_workspace()` if the value must reach workspace state
5. update the matching English and Chinese docs pages
6. update `README.md` if the quickstart, top-level navigation, or documentation checklist changed
7. run the retained smoke and phantom regression suite when needed

## Add A New GUI Parameter
1. update the relevant config module under `configs/`
2. update `autoflow/config.py`
3. update the matching dataclass in `autoflow/core/models.py` if it belongs in workspace state
4. update UI construction in `autoflow/ui/app.py`
5. update UI-to-workspace and workspace-to-UI sync code in `autoflow/ui/app.py`
6. update docs in `docs/en/user/gui.md`, `docs/zh/user/gui.md`, and the relevant feature page
7. run the retained smoke and phantom regression suite when needed

## Add A New Pipeline Step
1. define or extend the step identifier in `autoflow/core/models.py`
2. wire the step handler in `autoflow/core/pipeline.py`
3. update `autoflow/processing.py` if the batch order changes
4. update `autoflow/ui/app.py` if the GUI exposes the step
5. update docs for step order, outputs, and usage
6. update `README.md` if the user-facing quickstart or top-level docs index changed
7. run the retained smoke and phantom regression suite when needed

## Add A New Output File
1. write the file in `autoflow/core/pipeline.py`, `autoflow/processing.py`, or `autoflow/rendering/videos.py`
2. update reporting code if summaries depend on it
3. document the file in `docs/en/user/outputs.md` and `docs/zh/user/outputs.md`
4. update `README.md` if the quickstart-visible outputs changed
5. run the retained smoke and phantom regression suite when needed

## Add Or Change Loader Behavior
1. edit `autoflow/algorithms/data.py` or `autoflow/algorithms/dicom.py`
2. keep the `LoadedCase` contract consistent in `autoflow/case_types.py`
3. update `autoflow/core/pipeline.py` only if downstream workspace mapping changes
4. update the English and Chinese input docs
5. run the retained smoke and phantom regression suite when needed

## Add Or Change Segmentation Behavior
1. edit source logic in `autoflow/algorithms/segmentation.py`
2. edit workspace state in `autoflow/core/models.py` if needed
3. edit GUI wiring in `autoflow/ui/app.py` and `autoflow/ui/segmentation.py`
4. edit CLI wiring in `autoflow/cli.py` and `autoflow/api.py` if batch behavior changes
5. update segmentation docs in both languages
6. run the retained smoke and phantom regression suite when needed
