# Windows CI GUI verification

The portable build verifies the GUI with `PYVISTA_OFF_SCREEN=true`,
`QT_QPA_PLATFORM=offscreen` and the Mesa `vtkOSOpenGLRenderWindow` backend.
PyVistaQt intentionally sets `QtInteractor.iren` to `None` in this mode.
`SceneController` skips mouse-picking observer registration when there is no
interactor; rendering, phantom loading and the QC HTML checks still run.
Interactive desktop windows retain their normal picking observers.

The regression is covered by
`test_native_offscreen_gui_phantom_starts_without_mouse_interactor` in the
retained `tests/test_smoke_phantoms.py` file. It constructs the native Qt main
window with offscreen mode enabled, without the Linux SSH renderer. The portable
builder separately checks nonempty rendered images and the QC page through
`packaging/windows/portable_check.py --gui`.

`build-check.json` records `off_screen`, `interactor_available` and
`render_window` in `checks.gui_render`. A successful offscreen check can have
`interactor_available=false`; an interactive desktop check requires an interactor.
GitHub-hosted Windows verification does not establish GPU execution. Use
`--gui --require-gpu` on a Windows GPU workstation for that qualification.

Code owners: `autoflow/ui/viewer.py` for observer registration,
`packaging/windows/portable_check.py` for build diagnostics, and
`.github/workflows/windows-portable.yml` for the CI rendering environment.
