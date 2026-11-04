"""Rebuild the bundled VTP markers; the penguin geometry is the original model."""

import numpy as np
import pyvista as pv
from vtkmodules.vtkRenderingCore import vtkPropAssembly
from pathlib import Path
from vtkmodules.vtkFiltersCore import vtkAppendPolyData


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


def rabbit():
    """Round, matte rabbit in the same LR/AP/FH frame as the penguin."""
    assembly = vtkPropAssembly()

    def add(mesh, color):
        actor = pv.Actor(mapper=pv.DataSetMapper(dataset=mesh))
        actor.prop.color = color
        assembly.AddPart(actor)

    def ellipsoid(center, radii, color, tilt=0):
        mesh = pv.Sphere(theta_resolution=40, phi_resolution=32)
        mesh.points *= 2*np.asarray(radii)
        mesh.points += np.asarray(center)
        if tilt:
            mesh.rotate_y(tilt, point=center, inplace=True)
        mesh.compute_normals(inplace=True)
        add(mesh, color)

    fur, belly = '#f1e9dd', '#fff8ed'
    pink, blush, ink = '#eab7c2', '#edc5c3', '#34313c'
    # Short pear-shaped torso and a generous round head, without a long neck.
    ellipsoid((0, .035, -.43), (.60, .43, .62), fur)
    ellipsoid((0, -.370, -.43), (.43, .095, .43), belly)
    ellipsoid((0, .43, -.68), (.22, .22, .22), belly)
    # Slightly splayed, rounded ears. The inner pink pieces sit on their fronts.
    for side in (-1, 1):
        z = 1.06 if side < 0 else 1.10
        tilt = side*11 if side < 0 else side*17
        ellipsoid((side*.29, .035, z), (.185, .145, .54), fur, tilt)
        ellipsoid((side*.29, -.091, z+.015), (.105, .035, .40), pink, tilt)
    ellipsoid((0, 0, .35), (.69, .47, .57), fur)
    for side in (-1, 1):
        ellipsoid((side*.59, .015, -.40), (.135, .16, .29), fur, -side*20)
        ellipsoid((side*.35, -.20, -.98), (.25, .32, .135), fur)
        # Small inset eyes and tiny highlights keep the face soft at icon sizes.
        ellipsoid((side*.24, -.438, .40), (.063, .027, .082), ink)
        ellipsoid((side*.24-.018, -.462, .425), (.016, .010, .020), '#fffdf8')
        ellipsoid((side*.39, -.397, .20), (.095, .021, .045), blush)
        ellipsoid((side*.091, -.449, .175), (.12, .038, .095), belly)
    ellipsoid((0, -.493, .235), (.054, .030, .039), '#d995a5')
    # Tiny curved mouth, no whisker clutter.
    for side in (-1, 1):
        t = np.linspace(0, 1, 20)
        x = side*.075*t
        z = .19-.045*np.sin(np.pi*t)
        mouth = pv.Spline(np.c_[x, np.full_like(x, -.491), z], 28).tube(radius=.006, n_sides=8)
        add(mouth, '#8e717b')
    # The same left-chest badge as the penguin supplies the anatomical cue.
    t = np.linspace(0, 2*np.pi, 80, endpoint=False)
    x = 16*np.sin(t)**3*.008-.19
    z = (13*np.cos(t)-5*np.cos(2*t)-2*np.cos(3*t)-np.cos(4*t))*.008-.33
    y = -.370-.095*np.sqrt(np.clip(1-(x/.43)**2-((z+.43)/.43)**2, 0, 1))-.010
    heart = pv.PolyData(np.c_[x, y, z], np.r_[len(t), np.arange(len(t))])
    add(heart.triangulate().extrude((0, -.020, 0), capping=True), '#e64852')
    return assembly


def save_assembly(assembly, path):
    """One portable PolyData; preserve original part order, colours and normals."""
    meshes, normals_present, sizes = [], [], []
    parts = assembly.GetParts()
    parts.InitTraversal()
    for index in range(parts.GetNumberOfItems()):
        actor = parts.GetNextProp()
        mesh = pv.wrap(actor.GetMapper().GetInput()).copy()
        normals_present.append(int('Normals' in mesh.point_data))
        if not normals_present[-1]:
            mesh.point_data['Normals'] = np.zeros((mesh.n_points, 3), dtype=np.float32)
        color = np.rint(np.asarray(actor.GetProperty().GetColor())*255).astype(np.uint8)
        mesh.point_data['RGB'] = np.tile(color, (mesh.n_points, 1))
        mesh.cell_data['part_id'] = np.full(mesh.n_cells, index, dtype=np.int32)
        meshes.append(mesh)
        sizes.append([mesh.n_points, mesh.verts.size, mesh.lines.size, mesh.faces.size, mesh.strips.size])
    append = vtkAppendPolyData()
    for mesh in meshes:
        append.AddInputData(mesh)
    append.Update()
    mesh = pv.wrap(append.GetOutput())
    mesh.field_data['part_sizes'] = np.asarray(sizes, dtype=np.int64)
    mesh.field_data['part_has_normals'] = np.asarray(normals_present, dtype=np.uint8)
    mesh.field_data['autoflow_orientation_asset'] = np.array([1], dtype=np.int32)
    mesh.save(path)
    return mesh


if __name__ == '__main__':
    destination = Path(__file__).resolve().parents[1] / 'autoflow/ui/assets/orientation'
    destination.mkdir(parents=True, exist_ok=True)
    save_assembly(anatomical_marker((1, 1, 1)), destination / 'penguin.vtp')
    save_assembly(rabbit(), destination / 'rabbit.vtp')
