from pymol import cmd

def count_hydrogen_bonds(pdb_file, nanobody_selection, protein_selection):
    # Load the PDB file
    cmd.load(pdb_file, 'complex')

    # Select the nanobody and protein parts
    cmd.select('nanobody', nanobody_selection)
    cmd.select('protein', protein_selection)

    # Find hydrogen bonds
    cmd.find_pairs('nanobody and name N,O', 'protein and name N,O', mode=2, cutoff=3.5)

    # Count hydrogen bonds
    num_hbonds = cmd.count_atoms('sele')
    print(f'Number of hydrogen bonds: {num_hbonds}')

    return num_hbonds

if __name__ == "__main__":
    pdb_file = 'path_to_your_pdb_file.pdb'  # Replace with your PDB file path
    nanobody_selection = 'chain A'  # Replace with your nanobody selection criteria
    protein_selection = 'chain B'  # Replace with your protein selection criteria

    cmd.reinitialize()
    count_hydrogen_bonds(pdb_file, nanobody_selection, protein_selection)
