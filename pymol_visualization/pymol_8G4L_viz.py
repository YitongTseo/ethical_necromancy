# pymol -cq pymol_8G4L_viz.py

# TO TAKE NICE IMAGES:
# 1. ray 2400, 2400
# 2. png nice_pic, dpi=300

import random
random.seed(42)  
# List of PDB files (add your paths here)
pdb_files = [
    "/Users/yitongtseo/Documents/GitHub/ethical_necromancy/pymol_visualization/8G4L/8g4l-pdb-bundle1.pdb",
    "/Users/yitongtseo/Documents/GitHub/ethical_necromancy/pymol_visualization/8G4L/8g4l-pdb-bundle2.pdb",
    "/Users/yitongtseo/Documents/GitHub/ethical_necromancy/pymol_visualization/8G4L/8g4l-pdb-bundle3.pdb",
    "/Users/yitongtseo/Documents/GitHub/ethical_necromancy/pymol_visualization/8G4L/8g4l-pdb-bundle4.pdb",
    "/Users/yitongtseo/Documents/GitHub/ethical_necromancy/pymol_visualization/8G4L/8g4l-pdb-bundle5.pdb",
    "/Users/yitongtseo/Documents/GitHub/ethical_necromancy/pymol_visualization/8G4L/8g4l-pdb-bundle6.pdb",
    "/Users/yitongtseo/Documents/GitHub/ethical_necromancy/pymol_visualization/8G4L/8g4l-pdb-bundle7.pdb",
    "/Users/yitongtseo/Documents/GitHub/ethical_necromancy/pymol_visualization/8G4L/8g4l-pdb-bundle8.pdb",
    "/Users/yitongtseo/Documents/GitHub/ethical_necromancy/pymol_visualization/8G4L/8g4l-pdb-bundle9.pdb",
]
# Define target sequence
# Full sequence: MGDSEMAVFGAAAPYLRKSEKERLEAQTRPFDLKKDVFVPDDKQEFVKAKIVSREGGKVTAETEYGKTVTVKEDQVMQQNPPKFDKIEDMAMLTFLHEPAVLYNLKDRYGSWMIYTYSGLFCVTVNPYKWLPVYTPEVVAAYRGKKRSEAPPHIFSISDNAYQYMLTDRENQSILITGESGAGKTVNTKRVIQYFAVIAAIGDRSKKDQSPGKGTLEDQIIQANPALEAFGNAKTVRNDNSSRFGKFIRIHFGATGKLASADIETYLLEKSRVIFQLKAERDYHIFYQILSNKKPELLDMLLITNNPYDYAFISQGETTVASIDDAEELMATDNAFDVLGFTSEEKNSMYKLTGAIMHFGNMKFKLKQREEQAEPDGTEEADKSAYLMGLNSADLLKGLCHPRVKVGNEYVTKGQNVQQVIYATGALAKAVYERMFNWMVTRINATLETKQPRQYFIGVLDIAGFEIFDFNSFEQLCINFTNEKLQQFFNHHMFVLEQEEYKKEGIEWTFIDFGMDLQACIDLIEKPMGIMSILEEECMFPKATDMTFKAKLFDNHLGKSANFQKPRNIKGKPEAHFSLIHYAGIVDYNIIGWLQKNKDPLNETVVGLYQKSSLKLLSTLFANYAGADAPIEKGKGKAKKGSSFQTVSALHRENLNKLMTNLRSTHPHFVRCIIPNETKSPGVMDNPLVMHQLRCNGVLEGIRICRKGFPNRILYGDFRQRYRILNPAAIPEGQFIDSRKGAEKLLSSLDIDHNQYKFGHTKVFFKAGLLGLLEEMRDERLSRIITRIQAQSRGVLARMEYKKLLERRDSLLVIQWNIRAFMGVKNWPWMKLYFKIKPLLK"
target_sequence_beginning = "SEMAVFGAAAPYLRK"
target_sequence_end = "NIRAFMGVKNWPWMKLYFKIKPLLK"

SPHERE_DIAMETER = 200 #200
binding_residues = [
    40,
    50,
]  # From just inspecting the pymol file I decided the 40 to 50 residues were a good spot...

# Load all PDB files
for pdb in pdb_files:
    print("loading ... ", pdb)
    cmd.load(pdb)

# Color everything gray
cmd.color("gray", "all")


ref_coord = [434.96356821, 434.87994265, 434.87994265]
if ref_coord is None:
    all_coords = []
    for obj in cmd.get_object_list():
        # Get model chains
        chains = cmd.get_chains(obj)
        for chain in chains:
            model = cmd.get_model(f"{obj} and chain {chain}")
            idx = 0
            skip = 100
            for atom in model.atom:
                if idx >= skip:
                    all_coords.append(model.atom[0].coord)
                    idx = 0
                else:
                    idx += 1

    import numpy as np

    ref_coord = np.mean(all_coords, axis=0)

cmd.pseudoatom(f"center_1", pos=[434.96356821, 434.87994265, 434.87994265], elem="Au")
cmd.pseudoatom(f"center_bad", pos=[0, 0, 0], elem="Au")


# Load the nanobody PDB file (do this outside your loop, once)
cmd.load(
    "/Users/yitongtseo/Documents/GitHub/ethical_necromancy/pymol_visualization/6ysy_scaffold_perhaps.pdb",
    "nanobody",
)

sphere_count = 0


def create_sphere(position, diameter):
    global sphere_count
    # Create a pseudoatom at the specified position
    cmd.pseudoatom(
        f"big_sphere_{sphere_count}", pos=position, elem="C"
    )  # Using carbon as a placeholder
    cmd.color("pink", f"big_sphere_{sphere_count}")
    # Set the sphere size
    cmd.set("sphere_scale", diameter / 2)  # Adjust scale for visual representation

    # Show the sphere
    cmd.show("spheres", f"big_sphere_{sphere_count}")
    print("created sphere ", f"big_sphere_{sphere_count}")
    sphere_count += 1


print("whats ref_coord? ", ref_coord)
for obj in cmd.get_object_list():
    # Get model chains
    chains = cmd.get_chains(obj)

    for chain in chains:
        seq = cmd.get_fastastr(f"{obj} and chain {chain}").replace("\n", "")
        start = 0

        while True:
            start = seq.find(target_sequence_beginning, start)

            if start == -1:
                break

            end = seq.find(target_sequence_end, start)
            if end == -1:
                start += 1
                continue

            print(f"Hit in chain {chain}! @ {start} ... to {end}")
            # if random.random() > 0.3421:
            #     print(' but boo hoo, we are skipping!')
            #     continue

            cmd.color(
                "red",
                f"{obj} and chain {chain} and resi {start + 1}-{end + len(target_sequence_end)}",
            )

            # Create and translate the nanobody
            cmd.create(f"{obj}_nanobody_{chain}_{start}", "nanobody")
            cmd.align(
                f"{obj}_nanobody_{chain}_{start}",
                f"{obj} and chain {chain} and resi {start + 1}-{end + len(target_sequence_end)}",
            )
            cmd.color("pink", f"{obj}_nanobody_{chain}_{start}")
            cmd.remove(f"{obj}_nanobody_{chain}_{start} and resi 696-1470")

            # 40 to 50 is a good target...
            # Add a sphere representing the conducting polymer...
            model = cmd.get_model(
                f"{obj} and chain {chain} and resi {binding_residues[0]}"
            )
            if model.atom:
                first_residue_coord = model.atom[
                    0
                ].coord  # Access the first atom's coordinates
            else:
                raise ValueError("No atoms found for the specified residue.")

            model = cmd.get_model(
                f"{obj} and chain {chain} and resi {binding_residues[1]}"
            )
            if model.atom:
                last_residue_coord = model.atom[
                    0
                ].coord  # Access the first atom's coordinates
            else:
                raise ValueError("No atoms found for the specified residue.")

            center_coord = [
                (first_residue_coord[i] + last_residue_coord[i]) / 2 for i in range(3)
            ]

            # Calculate the direction vector
            direction_vector = [ref_coord[i] - center_coord[i] for i in range(3)]
            # Normalize the direction vector
            norm = sum(d**2 for d in direction_vector) ** 0.5
            direction_vector = [d / norm for d in direction_vector]

            if SPHERE_DIAMETER is not None:
                # Scale the direction vector by the desired distance (10 units)
                translation_distance = -SPHERE_DIAMETER / 2  # Adjust the distance as needed
                translation_vector = [d * translation_distance for d in direction_vector]

                # Calculate the new position
                new_position = [center_coord[i] + translation_vector[i] for i in range(3)]

                create_sphere(new_position, SPHERE_DIAMETER)

            start += 1


# Save the combined modified structure as a PyMOL session
cmd.save(
    "/Users/yitongtseo/Documents/GitHub/ethical_necromancy/pymol_visualization/combined_8G4L.pse"
)


cmd.set_view ([-0.190689877,    0.133922279,   -0.972473383,\
    -0.980345786,   -0.077024817,    0.181627214,\
    -0.050581064,    0.987993538,    0.145976976,\
     0.000465725,    0.000289679, -2489.378906250,\
   434.201965332,  432.063110352,  405.747375488,\
  1720.316406250, 3258.409912109,  -19.999998093] )


image_filename = "/Users/yitongtseo/Documents/GitHub/ethical_necromancy/pymol_visualization/20nm_side_view.png"

print('gathering side view')
cmd.png(image_filename, width=2000, height=2000, dpi=300)


cmd.set_view ([1.000000000,    0.000000000,    0.000000000,\
     0.000000000,    1.000000000,    0.000000000,\
     0.000000000,    0.000000000,    1.000000000,\
     0.000000000,    0.000000000, -1893.886230469,\
   431.167541504,  431.326904297,  386.559692383,\
  1338.857177734, 2448.914306641,  -19.999998093] )
print('gathering top view')

image_filename = "/Users/yitongtseo/Documents/GitHub/ethical_necromancy/pymol_visualization/20nm_top_view.png"
cmd.png(image_filename, width=2000, height=2000, dpi=300)
