from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pickle

# Parameters
COEFFICIENT = 1.0
spacing = 2 * 4 * COEFFICIENT
cone_radius = 4 * COEFFICIENT # FYI: This cannot be bigger than spacing otherwise we get a kernel crash...
assert cone_radius <= spacing, "cone_radius must be less than or equal to spacing to avoid kernel crash"
cone_depth = 4.0 * COEFFICIENT

RAISED_DEPTH = 0  # cone_depth / 2
connecting_cylinder_radius = 0.75 * COEFFICIENT
connecting_cylinder_depth = cone_depth  # 0.5 * cone_depth
sphere_radius = 1.5 * COEFFICIENT
sphere_depth = connecting_cylinder_depth * 1.5  # (cone_depth + (sphere_radius / 4))


def build_divot_geometry():
    cone = (
        cq.Workplane("XY")
        .circle(cone_radius)
        .workplane(offset=-cone_depth)
        .circle(0.0001)
        .loft()
        .translate((0, 0, RAISED_DEPTH))
    )

    top_cylinder = (
        cq.Workplane("XY")
        .cylinder(cone_depth * 4, cone_radius)
        .translate((0, 0, cone_depth * 2 + RAISED_DEPTH))
    )

    connecting_cylinder = (
        cq.Workplane("XY")
        .cylinder(connecting_cylinder_depth, connecting_cylinder_radius)
        .translate((0, 0, RAISED_DEPTH - connecting_cylinder_depth))
    )

    sphere = (
        cq.Workplane("XY")
        .sphere(sphere_radius)
        .translate((0, 0, RAISED_DEPTH - sphere_depth))
    )

    return cone.union(top_cylinder).union(connecting_cylinder).union(sphere)


def get_z_top(mesh_bytes, x, y):
    mesh = pickle.loads(mesh_bytes)
    ray_origins = np.array([[x, y, 100]])
    ray_directions = np.array([[0, 0, -1]])
    locations, _, _ = mesh.ray.intersects_location(ray_origins, ray_directions)
    return locations[0][2] if len(locations) > 0 else None

def create_translated_divot(args):
    x, y, mesh_bytes = args
    try:
        mesh = pickle.loads(mesh_bytes)
        ray_origins = np.array([[x, y, 100]])
        ray_directions = np.array([[0, 0, -1]])
        locations, _, _ = mesh.ray.intersects_location(ray_origins, ray_directions)
        if len(locations) == 0:
            return None
        z_top = locations[0][2]
        divot = build_divot_geometry()
        return divot.translate((x, y, z_top))
    except Exception as e:
        print(f"Failed divot at ({x}, {y}): {e}")
        return None
