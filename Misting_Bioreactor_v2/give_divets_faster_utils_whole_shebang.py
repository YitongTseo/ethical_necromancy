import cadquery as cq
import trimesh
import numpy as np
import pickle
import os
from tqdm import tqdm
import random

# Note: Make sure the 'rtree' package is installed for trimesh's ray casting to work:
# pip install rtree

# Parameters
COEFFICIENT = 1.0
SPACING = 3 * 4 * COEFFICIENT
cone_radius = 4 * COEFFICIENT
cone_depth = 4.0 * COEFFICIENT

RAISED_DEPTH = 0  # cone_depth / 2
connecting_cylinder_radius = 0.75 * COEFFICIENT
connecting_cylinder_depth = cone_depth
sphere_radius = 1.5 * COEFFICIENT
sphere_depth = connecting_cylinder_depth * 1.5

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

def create_translated_divot(x, y, mesh, num_rays=1, dither_amount=0.00, min_z_threshold=0.1):
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
        if loc[2] > min_z_threshold:  # Only consider intersections above a certain Z
            highest_z = max(highest_z, loc[2])
            valid_intersections = True

    if not valid_intersections:
        return None

    divot = build_divot_geometry()
    return divot.translate((x, y, highest_z))
    # except Exception as e:
    #     print(f"Failed divot at ({x:.2f}, {y:.2f}): {e}")
    #     return None

if __name__ == "__main__":
    # Load your sloped STEP file
    model = cq.importers.importStep("face v1.step")

    # Get bounding box
    bbox = model.val().BoundingBox()
    xmin, xmax = bbox.xmin, bbox.xmax
    ymin, ymax = bbox.ymin, bbox.ymax
    zmin = bbox.zmin

    print(f"Model bounds: X({xmin:.2f}, {xmax:.2f}), Y({ymin:.2f}, {ymax:.2f})")

    # Convert CQ model to mesh for raycasting
    if os.path.exists('face_mesh.glb'):
        print('Loading existing mesh...')
        mesh = trimesh.load("face_mesh.glb")

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
        vertices, faces = model.val().tessellate(0.1)  # Using finer tessellation
        vertices_np = np.array([[v.x, v.y, v.z] for v in vertices])
        faces_np = np.array(faces)
        mesh = trimesh.Trimesh(vertices=vertices_np, faces=faces_np)
        mesh.export("face_mesh.glb")

    print(f"Mesh has {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")

    # # Test a ray cast to verify the mesh works
    # test_x, test_y = (xmin + xmax) / 2, (ymin + ymax) / 2
    # test_ray_origin = np.array([[test_x, test_y, 100]])
    # test_ray_direction = np.array([[0, 0, -1]])

    # test_locations, _, _ = mesh.ray.intersects_location(test_ray_origin, test_ray_direction)
    # if len(test_locations) > 0:
    #     print(f"Test ray cast successful at ({test_x:.2f}, {test_y:.2f}). Z-top: {test_locations[0][2]:.2f}")
    # else:
    #     print(f"Warning: Test ray cast failed at ({test_x:.2f}, {test_y:.2f}). Check your mesh.")

    mesh_bytes = pickle.dumps(mesh)

    # Use fewer divots initially to test the process
    # step_size = int(SPACING)

    # Let's first visualize where the model actually is
    # test_points = []
    # for x in np.linspace(precise_xmin, precise_xmax, num=10):
    #     for y in np.linspace(precise_ymin, precise_ymax, num=10):
    #         ray_origin = np.array([[x, y, 100]])
    #         ray_direction = np.array([[0, 0, -1]])
    #         locations, _, _ = mesh.ray.intersects_location(ray_origin, ray_direction)
    #         if len(locations) > 0:
    #             test_points.append((x, y, locations[0][2]))

    # print(f"Found {len(test_points)} test points with intersections")
    # if test_points:
    #     print("Sample successful intersections:")
    #     for i, (x, y, z) in enumerate(test_points[:5]):
    #         print(f"  Point {i+1}: ({x:.2f}, {y:.2f}, {z:.2f})")

    all_divots = []
    for x_idx, x in tqdm(enumerate(range(int(xmin), int(xmax) + 2, int(SPACING))), total=int((xmax-xmin) / SPACING)):
        for y in range(int(ymin), int(ymax) + 2, int(SPACING)):
            divot = create_translated_divot(float(x), float(y), mesh)
            if divot is not None:
                all_divots.append(divot)

    print(f"Successfully created {len(all_divots)} divots")

    if not all_divots:
        print("No divots were created. Exiting.")
        exit(1)

    # Create a single divot as a base and union the rest
    divots = all_divots[0]

    # Unite divots in batches to avoid geometry errors
    batch_size = 10
    for i in range(1, len(all_divots), batch_size):
        batch = all_divots[i:i+batch_size]
        try:
            # Combine divots in this batch
            batch_union = batch[0]
            for divot in batch[1:]:
                batch_union = batch_union.union(divot)

            # Add this batch to the main divots
            divots = divots.union(batch_union)
            print(f"Processed batch {i//batch_size + 1}/{(len(all_divots)-1)//batch_size + 1}")
        except Exception as e:
            print(f"Error in batch {i//batch_size + 1}: {e}")
            continue

    # # Save the divots for inspection
    try:
        cq.exporters.export(divots, "all_divots.step")
        print("Exported divots to all_divots.step")
    except Exception as e:
        print(f"Failed to export divots: {e}")



    # modified = model
    # batch_size = 20
    # for i in tqdm(range(0, len(all_divots), batch_size), desc="Cutting divot batches"):
    #     batch = all_divots[i:i + batch_size]
    #     if not batch:
    #         continue
    #     try:
    #         combined_divots = batch[0]
    #         for j in range(1, len(batch)):
    #             combined_divots = combined_divots.union(batch[j])
    #         modified = modified.cut(combined_divots)
    #     except Exception as e:
    #         print(f"Error cutting batch {i // batch_size + 1}: {e}")

    # # Export result
    # try:
    #     cq.exporters.export(modified, "divotted_face_v3.step")
    #     print("Successfully created and exported divotted model!")
    # except Exception as e:
    #     print(f"Failed to create final model: {e}")

