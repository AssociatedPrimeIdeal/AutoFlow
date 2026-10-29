"""Camera-linked anatomical marker and six standard anatomical views."""

import numpy as np
import pyvista as pv
from vtkmodules.vtkRenderingCore import vtkPropAssembly


def anatomical_marker(signs):
    """Penguin in LR/AP/FH axes, with a heart on the patient's left/front."""
    assembly = vtkPropAssembly()

    def add(mesh, color):
        actor = pv.Actor(mapper=pv.DataSetMapper(dataset=mesh))
        actor.GetProperty().SetColor(pv.Color(color).float_rgb)
        actor.GetProperty().SetAmbient(.35)
        actor.GetProperty().SetDiffuse(.65)
        actor.GetProperty().SetSpecular(.04)
        actor.GetProperty().SetInterpolationToPhong()
        actor.SetScale(*map(float, signs))
        assembly.AddPart(actor)

    def ellipsoid(center, radii, color, tilt=0):
        mesh = pv.Sphere(theta_resolution=24, phi_resolution=20)
        mesh.points *= 2*np.asarray(radii)
        mesh.points += np.asarray(center)
        if tilt:
            mesh.rotate_y(tilt, point=center, inplace=True)
        mesh.compute_normals(inplace=True)
        add(mesh, color)

    charcoal, flipper = "#2d3741", "#26313b"
    ivory, yellow, red = "#f4f3eb", "#f6cf36", "#e64852"
    # Broad, overlapping head/body volumes make one short, rounded silhouette.
    ellipsoid((0, .02, -.30), (.79, .54, .76), charcoal)
    ellipsoid((0, -.47, -.43), (.67, .15, .62), ivory)
    ellipsoid((0, 0, .45), (.73, .47, .70), charcoal)
    # The belly, projecting beak and forward feet identify the anterior side.
    # A small rear tail keeps the posterior silhouette distinct in profile.
    ellipsoid((0, .48, -.72), (.23, .28, .14), flipper)
    for side in (-1, 1):
        ellipsoid((side*.77, .02, -.10), (.14, .16, .32), flipper, tilt=-side*55)
        ellipsoid((side*.43, -.22, -1.00), (.27, .39, .12), yellow)
        eye_z = .64 if side < 0 else .61
        ellipsoid((side*.27, -.430, eye_z), (.195, .085, .205), ivory)
        # Small round pupils without highlights give the reference's blank stare.
        pupil_x = side*.27 + (.018 if side < 0 else -.008)
        ellipsoid((pupil_x, -.512, eye_z+.005), (.043, .024, .043), "#161d25")
    ellipsoid((0, -.535, .285), (.30, .20, .12), yellow)
    # The left-chest badge preserves the existing patient-side cue. Curve it
    # over the white belly so it remains attached when the marker rotates.
    t = np.linspace(0, 2*np.pi, 80, endpoint=False)
    x = 16*np.sin(t)**3*.014-.27
    z = (13*np.cos(t)-5*np.cos(2*t)-2*np.cos(3*t)-np.cos(4*t))*.014-.27
    y = -.47-.15*np.sqrt(np.clip(1-(x/.67)**2-((z+.43)/.62)**2, 0, 1))-.014
    points = np.c_[x, y, z]
    heart = pv.PolyData(points, np.r_[len(points), np.arange(len(points))])
    add(heart.triangulate().extrude((0, -.030, 0), capping=True), red)
    return assembly


class AnatomicalOrientationMarker:
    VIEWS = (
        ("Anterior", (0, -1, 0), (0, 0, 1)),
        ("Posterior", (0, 1, 0), (0, 0, 1)),
        ("Left", (-1, 0, 0), (0, 0, 1)),
        ("Right", (1, 0, 0), (0, 0, 1)),
        ("Superior", (0, 0, 1), (0, -1, 0)),
        ("Inferior", (0, 0, -1), (0, 1, 0)),
    )

    def __init__(self, scene):
        self.scene = scene
        self.plotter = scene.plotter
        self._pressed = None
        self._dragged = False
        self._next_view = 0
        self._last_direction = None
        self.actor = anatomical_marker(scene._display_axis_signs)
        self.widget = self.plotter.add_orientation_widget(
            self.actor, interactive=False, viewport=(.01, .01, .28, .32))
        self.widget.GetRenderer().SetLayer(3)
        if hasattr(self.plotter, "setToolTip"):
            self.plotter.setToolTip("Penguin orientation marker: heart = patient left. Click to return to "
                                   "Anterior; click again for Posterior / Left / Right / Superior / Inferior.")

    def _camera_direction(self):
        camera = self.plotter.camera
        direction = np.asarray(camera.GetPosition())-camera.GetFocalPoint()
        direction /= max(np.linalg.norm(direction), 1e-12)
        up = np.asarray(camera.GetViewUp())
        up /= max(np.linalg.norm(up), 1e-12)
        return np.r_[direction, up]

    def snap_next(self):
        if self._last_direction is None or not np.allclose(
                self._camera_direction(), self._last_direction, atol=1e-5):
            self._next_view = 0
        name, direction, up = self.VIEWS[self._next_view]
        camera = self.plotter.camera
        center = np.asarray(camera.GetFocalPoint())
        distance = max(np.linalg.norm(np.asarray(camera.GetPosition())-center), 1.)
        signs = self.scene._display_axis_signs
        camera.SetPosition(*(center+np.asarray(direction)*signs*distance))
        camera.SetViewUp(*(np.asarray(up)*signs))
        camera.OrthogonalizeViewUp()
        self.plotter.renderer.ResetCameraClippingRange()
        self._last_direction = self._camera_direction()
        self._next_view = (self._next_view+1) % len(self.VIEWS)
        self.scene.request_render()
        return name

    def dispatch(self, event):
        iren = getattr(self.plotter.iren, "interactor", self.plotter.iren)
        point = np.asarray(iren.GetEventPosition(), dtype=float)
        if event == "press":
            if not self.scene._axes_shown:
                return False
            size = np.asarray(self.plotter.render_window.GetSize())
            viewport = np.asarray(self.widget.GetViewport()).reshape(2, 2)
            if not np.all((point >= viewport[0]*size) & (point <= viewport[1]*size)):
                return False
            self._pressed, self._dragged = point, False
            return True
        if self._pressed is None:
            return False
        self._dragged |= np.linalg.norm(point-self._pressed) > 6
        if event == "release":
            self._pressed = None
            if not self._dragged:
                self.snap_next()
        return True

    def close(self):
        self._pressed = None
        self.widget.SetEnabled(False)
        self.widget.SetInteractor(None)
