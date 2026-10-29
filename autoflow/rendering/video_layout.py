"""Persistent captions for offline movies, using the shared GUI text palette."""

import math

from .style import foreground_color, style_scalar_bar


class VideoOverlay:
    """Add a title/phase header and an export-progress footer to one plotter.

    Frame and phase indices passed to :meth:`update` are zero based. Displayed
    ordinals are one based. RR is read in milliseconds from the workspace;
    playback FPS and camera rotation never determine acquisition phase time.
    The export canvas is fixed, so pixel positions are resolved only once.
    """

    def __init__(self, plotter, workspace, title):
        self._plotter = plotter
        self.title = str(title)
        try:
            rr_ms = float(getattr(workspace, "rr", None))
        except (TypeError, ValueError, OverflowError):
            rr_ms = float("nan")
        self._rr_ms = rr_ms if math.isfinite(rr_ms) and rr_ms > 0 else None

        width, height = (int(value) for value in plotter.window_size)
        scale = min(1.0, width / 1600.0, height / 1200.0)
        padding = max(1, round(28 * scale))
        header_size = max(1, round(26 * scale))
        footer_size = max(1, round(18 * scale))
        color = foreground_color(plotter.background_color, workspace)
        self._header = self._add_text(
            plotter, (padding, height - padding), color, header_size, "video_header", top=True
        )
        self._footer = self._add_text(
            plotter, (padding, padding), color, footer_size, "video_footer", top=False
        )
        self._header_text = None
        self._footer_text = None

    @staticmethod
    def _add_text(plotter, position, color, font_size, name, *, top):
        actor = plotter.add_text(
            "", position=position, color=color, font_size=font_size,
            font="arial", viewport=False, name=name, render=False,
        )
        # Tuple coordinates create a vtkTextActor rather than a CornerAnnotation.
        # Set the VTK size explicitly: PyVista's font_size multiplier otherwise
        # varies from the pixel-positioned actor's apparent text size.
        actor.SetTextScaleModeToNone()
        prop = actor.GetTextProperty()
        prop.SetFontSize(font_size)
        prop.SetJustificationToLeft()
        if top:
            prop.SetVerticalJustificationToTop()
        else:
            prop.SetVerticalJustificationToBottom()
        return actor

    def update(self, *, frame_index, total_frames, phase_index=None,
               phase_count=None, azimuth_deg=None):
        """Update captions and style any new scalar bar without rendering."""
        for bar in self._plotter.scalar_bars.values():
            style_scalar_bar(bar)
        header = self.title
        if phase_index is not None and phase_count is not None and phase_count > 0:
            count = int(phase_count)
            phase = int(phase_index)
            width = max(2, len(str(count)))
            phase_text = f"Phase {phase + 1:0{width}d}/{count:0{width}d}"
            if self._rr_ms is not None:
                phase_text += f"  |  {phase * self._rr_ms / count:.0f} ms"
            header += "\n" + phase_text

        total = int(total_frames)
        width = max(3, len(str(total)))
        footer = f"Export frame {int(frame_index) + 1:0{width}d}/{total:0{width}d}"
        if azimuth_deg is not None:
            angle = float(azimuth_deg)
            if math.isfinite(angle):
                footer = f"Rotation {angle % 360.0:05.1f} deg  |  " + footer

        if header != self._header_text:
            self._header.SetInput(header)
            self._header_text = header
        if footer != self._footer_text:
            self._footer.SetInput(footer)
            self._footer_text = footer
