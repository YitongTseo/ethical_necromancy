import cadquery as cq
import trimesh
import numpy as np
import pickle
import os
from tqdm import tqdm
import random
from cadquery import Vector

# Note: Make sure the 'rtree' package is installed for trimesh's ray casting to work:
# pip install rtree

# Parameters
COEFFICIENT = 1.0
SPACING = 2.8 * 4 * COEFFICIENT
cone_radius = 4 * COEFFICIENT
cone_depth = 4.0 * COEFFICIENT

RAISED_DEPTH = 0  # cone_depth / 2
connecting_cylinder_radius = 0.75 * COEFFICIENT
connecting_cylinder_depth = cone_depth
sphere_radius = 1.5 * COEFFICIENT
sphere_depth = connecting_cylinder_depth * 1.5

# ALERT: BE VERY CAREFUL NOT TO PUT ANYTHING BELOW THE SURFACE 
# OTHERWISE IT WILL FAIL!!!
def build_geometry_individually_packaged(x, y, z, ZMIN):
    if z + RAISED_DEPTH - cone_depth <= ZMIN:
        return []
    if z + RAISED_DEPTH - 2 * connecting_cylinder_depth  <= ZMIN:
        return []
    if  z + RAISED_DEPTH - sphere_depth - sphere_radius <= ZMIN:
        return []

    cone = (
        cq.Workplane("XY")
        .circle(cone_radius)
        .workplane(offset=-cone_depth)
        .circle(0.01)
        .loft()
        .translate((x, y, z + RAISED_DEPTH))
        )

    # this top sphere replaces the original clone
    # top_sphere = (
    #     cq.Workplane("XY")
    #     .sphere(cone_depth)
    #     .translate((x, y, z + RAISED_DEPTH))
    # )

    top_cylinder = (
        cq.Workplane("XY")
        .cylinder(cone_depth * 4, cone_radius)
        .translate((x, y, z + cone_depth * 2 + RAISED_DEPTH))
    )

    connecting_cylinder = (
        cq.Workplane("XY")
        .cylinder(connecting_cylinder_depth, connecting_cylinder_radius)
        .translate((x, y, z + RAISED_DEPTH - connecting_cylinder_depth + 0.1))
    )

    sphere = (
        cq.Workplane("XY")
        .sphere(sphere_radius)
        .translate((x, y, z + RAISED_DEPTH - sphere_depth))
    )

    return [cone, top_cylinder, connecting_cylinder, sphere]

def create_translated_divot(x, y, mesh, ZMIN, num_rays=1, dither_amount=0.00):
    ray_origins = np.array([[x, y, 100]])
    ray_directions = np.array([[0, 0, -1]])
    
    # Check if we have a scene or a trimesh
    if isinstance(mesh, trimesh.Scene):
        # Get the geometry from the scene
        geom_keys = list(mesh.geometry.keys())
        if len(geom_keys) > 0:
            # Use the first mesh in the scene
            mesh = next(iter(mesh.geometry.values()))
        else:
            print(f"No geometry found in scene at ({x}, {y})")
            return None

    # Debug: Print every 10th test point for visibility
    debug = (int(x * 10) % 40 == 0 and int(y * 10) % 40 == 0)
    if debug:
        print(f"Testing divot at ({x:.2f}, {y:.2f})")
        
    ray_origins = []
    ray_directions = []

    for _ in range(num_rays):
        # Generate slightly dithered ray origins
        origin_x = x + random.uniform(-dither_amount, dither_amount)
        origin_y = y + random.uniform(-dither_amount, dither_amount)
        ray_origins.append([origin_x, origin_y, 200])  # Still casting downwards from a high Z

        ray_directions.append([0, 0, -1])
    ray_origins = np.array(ray_origins)
    ray_directions = np.array(ray_directions)


    locations, _, _ = mesh.ray.intersects_location(ray_origins, ray_directions)

    if len(locations) == 0:
        return None

    # Find the highest Z-top location
    highest_z = -float('inf')
    valid_intersections = False
    for loc in locations:
        if loc[2] > ZMIN:  # Only consider intersections above a certain Z
            highest_z = max(highest_z, loc[2])
            valid_intersections = True

    if not valid_intersections:
        return None

    return build_geometry_individually_packaged(x, y, highest_z, ZMIN)


if __name__ == "__main__":
    # Load your sloped STEP file
    model = cq.importers.importStep("bigger_baseline v2.step")

    # Get bounding box
    bbox = model.val().BoundingBox()
    xmin, xmax = bbox.xmin, bbox.xmax
    ymin, ymax = bbox.ymin, bbox.ymax
    ZMIN = bbox.zmin

    print(f"Model bounds: X({xmin:.2f}, {xmax:.2f}), Y({ymin:.2f}, {ymax:.2f}), zmin {ZMIN}")

    # Convert CQ model to mesh for raycasting
    if os.path.exists('bigger_baseline v2.glb'):
        print('Loading existing mesh...')
        mesh = trimesh.load("bigger_baseline v2.glb")

        # Fix: Ensure we have a Trimesh, not a Scene
        if isinstance(mesh, trimesh.Scene):
            # Try to extract the mesh from the scene
            if len(mesh.geometry) > 0:
                print('Converting Scene to Trimesh...')
                # Get the first mesh in the scene
                mesh = next(iter(mesh.geometry.values()))
            else:
                print('Error: Loaded file contains an empty scene')
                exit(1)
    else:
        print('Computing new mesh...')
        vertices, faces = model.val().tessellate(2)  # Using finer tessellation
        vertices_np = np.array([[v.x, v.y, v.z] for v in vertices])
        faces_np = np.array(faces)
        mesh = trimesh.Trimesh(vertices=vertices_np, faces=faces_np,)
        mesh.export("bigger_baseline v2.glb")

    print(f"Mesh has {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")

    mesh_bytes = pickle.dumps(mesh)

    all_divots = []

    for x_idx, x_start in tqdm(enumerate(range(int(xmin), int(xmax) + 2, int(SPACING))),
                        total=int((xmax - xmin) / SPACING)):
        for y_idx, y in enumerate(range(int(ymin), int(ymax) + 2, int(SPACING))):
            # Stagger every other row
            x_offset = SPACING / 2 if y_idx % 2 == 1 else 0
            x = x_start + x_offset

            divot = create_translated_divot(float(x), float(y), mesh, ZMIN=ZMIN)
            if divot is not None:
                all_divots.append(divot)

    # for x_idx, x in tqdm(enumerate(range(int(xmin), int(xmax) + 2, int(SPACING))), total=int((xmax-xmin) / SPACING)):
    #     for y in range(int(ymin), int(ymax) + 2, int(SPACING)):
    #         divot = create_translated_divot(float(x), float(y), mesh, ZMIN=ZMIN)
    #         if divot is not None:
    #             all_divots.append(divot)

    print(f"Successfully created {len(all_divots)} divots")

    if not all_divots:
        print("No divots were created. Exiting.")
        exit(1)

    # Unite divots in batches to avoid geometry errors
    batch_size = 100
    modified_model = model
    num_elements_in_tuple = len(all_divots[0]) if all_divots else 1  # Assuming all tuples have the same length

    for i in tqdm(range(0, len(all_divots), batch_size), desc="Cutting divot batches"):
        batch = all_divots[i:i + batch_size]
        if not batch:
            continue

        for element_index in range(num_elements_in_tuple):
            print(f'element {element_index}')
            current_element_batch = [divot_tuple[element_index] for divot_tuple in batch if len(divot_tuple) == num_elements_in_tuple]
            try:
                combined_divots = current_element_batch[0]
                for j in range(1, len(current_element_batch)):
                    combined_divots = combined_divots.union(current_element_batch[j])
                modified_model = modified_model.cut(combined_divots)
            except Exception as e:
                print(f"Error cutting batch {i // batch_size + 1}: {e}")

    # Export the final modified model
    try:
        cq.exporters.export(modified_model, "bigger_baseline_divotted_v5.step")
        print("Successfully created and exported the divotted model!")
    except Exception as e:
        print(f"Failed to export the final model: {e}")
