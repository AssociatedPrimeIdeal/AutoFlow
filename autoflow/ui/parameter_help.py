"""Short parameter labels with explanations available on hover."""

import html


FLOW_HELP = {
    'integration_backend': 'GPU preferred uses CuPy/CUDA RK4 when available and falls back to VTK on failure. CPU (VTK) forces the existing tracer. GPU (CUDA) reports an error if unavailable. This applies to instantaneous 3D streamlines, including movies; pathlines retain their separate integrator.',
    'seed_region': 'Choose the domain in which trajectories may be seeded and integrated. Full FOV uses finite velocity samples; segmentation uses the vessel mask; Noise removed region uses the noise-removal keep mask independently of segmentation.',
    'seed_distribution': 'Uniform grid places seeds on a physical grid. Random samples seeds within the selected domain. Grid spacing controls the uniform count; count/ratio settings apply only to Random.',
    'grid_spacing_mm': 'Physical spacing between uniform seeds, in millimetres. For 3D streamlines, 0 chooses three voxel spacings independently along each axis. Smaller spacing creates more trajectories and takes longer.',
    'seed_mode': 'For Random seeding, choose an explicit count or a fraction of eligible voxels. Uniform-grid counts are determined by spacing and support.',
    'seed_count': 'Number of random seeds requested in the selected domain. The actual supported count is displayed after generation.',
    'seed_ratio': 'Fraction of eligible voxels used as random seeds. The minimum seed count is also applied. This has no effect on uniform-grid seeding.',
    'min_seeds': 'Minimum requested random seed count when Ratio is selected, limited by the available domain.',
    'rng_seed': 'Reproducible random seed. Changing it selects different random start points; it has no effect on the uniform grid.',
    'integration_step_mm': 'Integration step measured in physical millimetres. Auto uses the tracer default. This is independent of the spacing between seeds. Smaller steps can improve tracing detail at increased cost.',
    'max_steps': 'Maximum integration steps for a trajectory. Increasing it can extend trajectories until another stopping condition is reached, at increased computation cost.',
    'terminal_speed': 'Stop integration when velocity falls below this threshold, in metres per second (m/s). This changes the generated trajectories. It differs from the display-only speed threshold in cm/s.',
    'display_style': 'Thin lines uses Thickness in screen pixels. Tubes uses Thickness as a physical circular tube radius in millimetres. Each style retains its own separate value.',
    'display_min_speed_cm_s': 'Hide visible segments below this speed, in centimetres per second (cm/s). Zero disables the filter. Existing trajectories are reused; integration is unchanged.',
    'line_width': 'Screen stroke thickness in pixels, used by Thin lines. A 1 px stroke stays approximately 1 pixel wide when zooming. It does not set a physical diameter. This setting is independent of Tube radius.',
    'tube_radius': 'Physical radius of the 3D circular tube surface, in millimetres, used only by Tubes. Radius 1 mm means diameter 2 mm. The tube grows on screen when the scene is zoomed in. This is a display setting, not vessel size or an integration parameter.',
    'pathline_seed_region': 'Seed time-dependent pathlines inside the segmentation or inside the selected plane contour. A contour is resolved separately for each launch phase.',
    'pathline_seed_distribution': 'Random seeds use count/ratio controls; Uniform grid uses physical grid spacing inside the selected plane domain.',
    'pathline_grid_spacing_mm': 'Physical spacing in millimetres between uniform plane seeds.',
    'pathline_integration_step_mm': 'Physical pathline integration step in millimetres, or auto. It is independent of the spacing between launch points.',
    'pathline_temporal_cache_mb': 'Memory budget in MiB for retained temporal velocity data used by pathline integration. It does not change the requested seed count.',
    'pathline_display_style': 'Show complete particle histories, current particles, or particles with short trails. This reuses the integrated paths.',
    'pathline_trail_duration_ms': 'Visible age of the short particle trail, in milliseconds. Integration results are unchanged.',
    'pathline_color_mode': 'Colour pathlines by plane, group, a uniform colour, velocity, or particle age. This changes display only.',
}

SHARED_HELP = {
    'min_edge_points': 'Minimum retained terminal-branch length in graph edges, not millimetres or a point-deletion quota. Default 9 removes arms with 1 to 8 edges and retains 9-edge arms; junction-to-junction links stay. Set 0 or 1 to disable. Regenerate Graph and Planes after changing this value.',
    'cmap': 'Colormap used to map the displayed scalar quantity to colour. It changes appearance only.',
    'opacity': 'Display opacity: 0 is transparent and 1 is fully opaque.',
    'range_mode': 'Choose automatic colour limits or a manually specified lower and upper limit. This changes the mapping to colours, not the data.',
    'low': 'Lower scalar colour limit. A manual upper limit must be greater than this value.',
    'high': 'Upper scalar colour limit. It must exceed the manual lower limit.',
    'color': 'Uniform display colour used when Fixed colour is selected.',
    'color_mode': 'Choose colouring by the displayed speed or a fixed colour.',
    'quantity': 'Total speed includes all velocity components; in-plane speed includes only components tangent to the displayed slice.',
    'visible': 'Show or hide this display layer without changing its source data.',
    'enabled': 'Enable this overlay. The required velocity, mask or generated trajectories must be available.',
    'line_width': 'Cosmetic stroke thickness in screen pixels.',
    'arrows': 'Show flow direction arrows along the displayed trajectories.',
    'arrow_size': 'Physical size of the direction arrow in millimetres.',
    'arrow_spacing': 'Physical distance in millimetres between direction arrows.',
    'min_length': 'Hide visible curve pieces shorter than this physical length. This does not regenerate 3D trajectories.',
    'density': 'Seed density of the 2D planar integration. Higher values reduce seed spacing; they do not increase the existing 3D seed count.',
    'fraction': 'Fraction of existing 3D trajectory pieces to display. Generate Streamlines first. Increasing beyond the existing seed count requires regeneration.',
    'slab': 'Full physical thickness of the slab centred on the slice, in millimetres. Only generated 3D segments inside this slab are projected.',
    'max_length': 'Maximum physical length in millimetres for each planar trace. Zero selects the automatic length.',
    'terminal_speed': 'Minimum speed for 2D integration, in centimetres per second (cm/s). It is independent of the 3D integration threshold in m/s.',
    'max_steps': 'Maximum number of integration steps in each direction for the 2D trace.',
    'background': 'Viewer background colour. It does not alter image intensities.',
    'zoom': 'Fraction of the full slice field of view shown initially. A value of 0.5 gives a view-only 2× zoom without cropping data.',
    'linked_zoom': 'Keep zoom and panning linked between the three slice views.',
    'order': 'Stack order of the three slice views. The selected measurement plane remains pinned first.',
    'crosshairs': 'Show the cursor guides. Editing centres and rotation handles are shown separately while plane editing is unlocked.',
    'orientation': 'Show anatomical direction labels along the visible edges of the slice.',
    'values': 'Show current voxel values and selected-plane measurement labels.',
    'plane_intersection': 'Show the selected plane intersection where applicable.',
    'crosshair_color': 'Colour used for cursor guides.',
    'plane_color': 'Colour used for the selected plane intersection guide.',
    'width': 'Display width in pixels of the colourbar.',
    'font_size': 'Font size of colourbar tick labels in points.',
    'smoothing': 'View-only contour smoothing. It does not change saved contours or numerical areas.',
    'minimum_points': 'Minimum point count for a displayed contour.',
    'snap_distance': 'Maximum endpoint-to-boundary snap distance for ROI editing, in millimetres. Zero selects the adaptive distance.',
    'labels': 'Comma-separated label IDs to display. Leave empty to display all available labels.',
    'smoothing_method': 'Radius adaptive fits a local quadratic over physical arc length, bounded to 0.15 local radii by default and kept within its vessel group. Endpoints and junction coordinates stay fixed. Legacy SG and no smoothing remain available.',
    'smoothing_radius_factor': 'Half-window in local lumen radii. Default 2 means samples on each side within 2r; it does not mean a fixed voxel window.',
    'smoothing_max_displacement_ratio': 'Maximum fitted-point movement divided by local radius. Default 0.15. The whole move also stays inside the original group lumen.',
    'smoothing_sample_step_mm': 'Uniform physical arc-length spacing before radius-adaptive fitting. Default 0.5 mm; independent of display interpolation density.',
    'smoothing_radius_sigma_mm': 'Gaussian smoothing scale of the local EDT radius along the path in mm. Default 3 mm stabilizes the fitting window.',
    'smoothing_backend': 'Auto uses batched CPU fits for small host-resident centerlines. GPU is optional and falls back to the same CPU fit if CUDA is unavailable. GPU transfer and launch overhead can make it slower.',
    'smoothing_window': 'Number of neighbouring samples for legacy Savitzky-Golay path smoothing only.',

    'smoothing_polyorder': 'Polynomial order used for path smoothing; it must be smaller than the smoothing window.',
    'inter_time': 'Path interpolation factor used before geometric processing.',
    'rho': 'Fluid density in kilograms per cubic metre (kg/m³). This affects derived physical quantities.',
    'viscosity': 'Dynamic fluid viscosity in millipascal seconds (mPa·s), used for wall shear stress.',
    'inward_distance': 'Physical inward probing distance in millimetres, or auto, for sampling the near-wall velocity profile.',
    'parabolic_fitting': 'Fit a parabolic near-wall velocity profile when estimating wall shear stress.',
    'no_slip_condition': 'Apply the no-slip wall condition during wall shear stress estimation.',
    'fov_boundary_filter': 'Exclude measurement planes whose ROI contacts the image field-of-view boundary beyond the configured perimeter fraction.',
    'fov_boundary_max_fraction': 'Maximum allowed fraction of ROI perimeter touching the image boundary. This is a perimeter ratio, not an area fraction.',
    'segmentation_filter': 'Require geometric segmentation support for the requested vessel or branch.',
    'scene_visible': 'Show the PWV plane layout in the 3D scene.',
}

INPUT_HELP = {
    'resolution': 'Voxel spacing in millimetres, ordered LR, AP, FH. Save keeps a draft; Apply and reload updates loaded data and dependent results without modifying the source metadata.',
    'venc': 'VENC values in cm/s, ordered LR, AP, FH. For phase-encoded input, VENC converts phase to velocity. For stored physical velocities, editing VENC does not rescale the stored values. Dual-VENC edits preserve low/high ratios. Apply and reload is required.',
    'spatial_order': 'Map stored spatial dimensions to LR/AP/FH. Reverse flips the corresponding array axis. Save keeps a draft; Apply and reload applies it to the loaded case.',
    'venc_order': 'Map stored velocity channels to LR/AP/FH and reverse velocity signs where needed. LR/RL means left/right, AP/PA front/back, and FH/HF feet/head. Apply and reload is required.',
}


def parameter_help(key, label, existing=''):
    leaf = key.rsplit('.', 1)[-1]
    if key.startswith('input.'):
        text = INPUT_HELP.get(leaf, label)
    elif key.startswith('streamline_params.'):
        fallback = leaf.removeprefix('pathline_')
        text = FLOW_HELP.get(leaf, FLOW_HELP.get(fallback, SHARED_HELP.get(leaf, label)))
    else:
        text = SHARED_HELP.get(leaf, label)
    if existing and existing not in text:
        text += '\n\n'+existing
    return text


def compact_label(label):
    return (label.replace('(mm; 0 = 3 × voxel spacing)', '(mm)')
            .replace('(mm or auto)', '(mm)').replace(' (0 = auto)', '')
            .replace(' (shared)', '').strip())


def bind_help(form, widget, text):
    tooltip = '<qt>'+html.escape(text).replace('\n', '<br>')+'</qt>'
    widget.setToolTip(tooltip)
    label = form.labelForField(widget)
    if label is not None:
        label.setToolTip(tooltip)
    return tooltip
