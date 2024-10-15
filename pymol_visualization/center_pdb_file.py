# pymol -cq center_pdb_file.py

# Load your PDB file
cmd.load("/Users/yitongtseo/Documents/GitHub/ethical_necromancy/pymol_visualization/canonacilized_3g9a_nanobody_VHH.pdb", "my_structure")

# Get all atoms' coordinates
all_atoms = cmd.get_model("my_structure").atom
all_coords = [atom.coord for atom in all_atoms]
# Calculate the centroid
centroid = [sum(coord) / len(all_atoms) for coord in all_coords]

# Translate all atoms to center them around (0, 0, 0)
cmd.translate([-centroid[0], -centroid[1], -centroid[2]], "my_structure")

# Optionally, save the modified structure
cmd.save("/Users/yitongtseo/Documents/GitHub/ethical_necromancy/pymol_visualization/centered_canonacilized_3g9a_nanobody_VHH.pdb", "my_structure")
