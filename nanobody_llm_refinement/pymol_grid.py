# Script to color and save pdb images
# pymol -cq pymol_grid.py <-- run in command line (from the conda base)

import os

pdb_files = [
    "/Users/yitongtseo/Documents/GitHub/ethical_necromancy/nanobody_llm_refinement/Haddock_results/MyoHead_0_A102T_None_cluster0.pdb",
    "/Users/yitongtseo/Documents/GitHub/ethical_necromancy/nanobody_llm_refinement/Haddock_results/MyoHead_0_A102V_None_cluster0.pdb",
    "/Users/yitongtseo/Documents/GitHub/ethical_necromancy/nanobody_llm_refinement/Haddock_results/MyoHead_0_L114M_A102H_cluster0.pdb",
    "/Users/yitongtseo/Documents/GitHub/ethical_necromancy/nanobody_llm_refinement/Haddock_results/MyoHead_0_A102T_L96A_cluster0.pdb",
    "/Users/yitongtseo/Documents/GitHub/ethical_necromancy/nanobody_llm_refinement/Haddock_results/MyoHead_0_A102V_N100D_cluster0.pdb",
    "/Users/yitongtseo/Documents/GitHub/ethical_necromancy/nanobody_llm_refinement/Haddock_results/MyoHead_0_N100D_L114M_cluster0.pdb",
    "/Users/yitongtseo/Documents/GitHub/ethical_necromancy/nanobody_llm_refinement/Haddock_results/MyoHead_0_A102V_L114M_cluster0.pdb",
    "/Users/yitongtseo/Documents/GitHub/ethical_necromancy/nanobody_llm_refinement/Haddock_results/MyoHead_0_L96A_L114M_cluster0.pdb",
    "/Users/yitongtseo/Documents/GitHub/ethical_necromancy/nanobody_llm_refinement/Haddock_results/MyoHead_0_A102I_L96A_cluster0.pdb",
    "/Users/yitongtseo/Documents/GitHub/ethical_necromancy/nanobody_llm_refinement/Haddock_results/MyoHead_0_L114M_None_cluster0.pdb",
    "/Users/yitongtseo/Documents/GitHub/ethical_necromancy/nanobody_llm_refinement/Haddock_results/MyoHead_0_A102H_None_cluster0.pdb",
    "/Users/yitongtseo/Documents/GitHub/ethical_necromancy/nanobody_llm_refinement/Haddock_results/MyoHead_0_N100D_A102T_cluster0.pdb",
    "/Users/yitongtseo/Documents/GitHub/ethical_necromancy/nanobody_llm_refinement/Haddock_results/MyoHead_0_None_None_cluster0.pdb",
    "/Users/yitongtseo/Documents/GitHub/ethical_necromancy/nanobody_llm_refinement/Haddock_results/MyoHead_0_A102V_L96A_cluster0.pdb",
    "/Users/yitongtseo/Documents/GitHub/ethical_necromancy/nanobody_llm_refinement/Haddock_results/MyoHead_0_N100D_A102R_cluster0.pdb",
    "/Users/yitongtseo/Documents/GitHub/ethical_necromancy/nanobody_llm_refinement/Haddock_results/MyoHead_0_A102I_L114M_cluster0.pdb",
    "/Users/yitongtseo/Documents/GitHub/ethical_necromancy/nanobody_llm_refinement/Haddock_results/MyoHead_0_L96A_A102H_cluster0.pdb",
    "/Users/yitongtseo/Documents/GitHub/ethical_necromancy/nanobody_llm_refinement/Haddock_results/MyoHead_0_N100D_A102H_cluster0.pdb",
    "/Users/yitongtseo/Documents/GitHub/ethical_necromancy/nanobody_llm_refinement/Haddock_results/MyoHead_0_A102R_L114M_cluster0.pdb",
    "/Users/yitongtseo/Documents/GitHub/ethical_necromancy/nanobody_llm_refinement/Haddock_results/MyoHead_0_N100D_A102I_cluster0.pdb",
    "/Users/yitongtseo/Documents/GitHub/ethical_necromancy/nanobody_llm_refinement/Haddock_results/MyoHead_0_N100D_L96A_cluster0.pdb",
    "/Users/yitongtseo/Documents/GitHub/ethical_necromancy/nanobody_llm_refinement/Haddock_results/MyoHead_0_A102R_L96A_cluster0.pdb",
    "/Users/yitongtseo/Documents/GitHub/ethical_necromancy/nanobody_llm_refinement/Haddock_results/MyoHead_0_A102T_L114M_cluster0.pdb",
    "/Users/yitongtseo/Documents/GitHub/ethical_necromancy/nanobody_llm_refinement/Haddock_results/MyoHead_0_N100D_None_cluster0.pdb",
    "/Users/yitongtseo/Documents/GitHub/ethical_necromancy/nanobody_llm_refinement/Haddock_results/MyoHead_0_A102I_None_cluster0.pdb",
    "/Users/yitongtseo/Documents/GitHub/ethical_necromancy/nanobody_llm_refinement/Haddock_results/MyoHead_0_L96A_None_cluster0.pdb",
    "/Users/yitongtseo/Documents/GitHub/ethical_necromancy/nanobody_llm_refinement/Haddock_results/MyoHead_0_A102R_None_cluster0.pdb",
]


for pdb_file in pdb_files:
    print("we in here?")
    # Load the PDB file
    cmd.load(pdb_file)

    # Color chains
    cmd.color("paleyellow", "chain A")
    cmd.color("wheat", "chain B")

    # Save an image
    image_filename = f"{pdb_file.replace('.pdb', '')}.png"
    cmd.png(image_filename, width=800, height=600, dpi=300)

    # Delete the object to load the next file
    cmd.delete("all")
