import pdb

# this may only work for the ones I created...
ROOT = "/Users/yitongtseo/Documents/GitHub/ethical_necromancy/nanobody_design_scratch/pdb_files"
RAW_FOLDER = "/home/yitongt/Documents/Github/ethical_necromancy/nanobody_design_scratch/get_results/esmfold_sanity_check"
PROCESSED_FOLDER = "/home/yitongt/Documents/Github/ethical_necromancy/nanobody_design_scratch/get_results/esmfold_sanity_check_canonaclized"
idxs_of_interest = [
    0
]

def find_glycine_bridge(lines, bridge_length=50):
    glycine_sequences = []
    current_sequence = []
    start_residue = None
    
    for line in lines:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            residue_name = line[17:20].strip()
            residue_id = int(line[22:26].strip())
            
            if residue_name == "GLY":
                if not current_sequence:
                    start_residue = residue_id
                current_sequence.append(line)
            else:
                if current_sequence:
                    if len(current_sequence) >= bridge_length:
                        glycine_sequences.append((start_residue, current_sequence))
                    current_sequence = []
    
    # Check the last sequence
    if current_sequence and len(current_sequence) >= bridge_length:
        glycine_sequences.append((start_residue, current_sequence))
    
    # Find the longest sequence of GLY
    longest_sequence = max(glycine_sequences, key=lambda x: len(x[1]), default=None)
    
    if longest_sequence:
        start_residue, sequence_lines = longest_sequence
        end_residue = start_residue + len(sequence_lines) - 1
        print(f"Found glycine bridge from residue {start_residue} to {end_residue}.")
        return start_residue, end_residue
    else:
        print("No glycine bridge of length 50 found.")
        return None, None

def split_pdb(input_file, output_file):
    with open(input_file, "r") as file:
        lines = file.readlines()
    
    bridge_start_residue, bridge_end_residue = find_glycine_bridge(lines)
    if bridge_start_residue is None or bridge_end_residue is None:
        print("Unable to split the PDB file due to missing bridge.")
        return
        
    chain1_lines, chain2_lines = [], []
    
    for line in lines:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            residue_name = line[17:20].strip()
            residue_id = int(line[22:26].strip())
            chain_id = line[21]
            
            # Skip bridge residues
            if residue_name == "GLY" and bridge_start_residue <= residue_id <= bridge_end_residue:
                continue
            
            # Change chain identifier after the bridge
            if residue_id > bridge_end_residue:
                chain_id = 'B'
            modified_line = line[:21] + chain_id + line[22:]
            
            if residue_id < bridge_start_residue:
                chain1_lines.append(line)
            elif residue_id > bridge_end_residue:
                chain2_lines.append(line)
        elif 'PARENT N/A' in line:
            continue
        else:
            chain1_lines.append(line)
            chain2_lines.append(line)

    with open(output_file.replace('.pdb', '_nanobody.pdb'), "w") as file:
        file.writelines(chain1_lines)
    
    with open(output_file.replace('.pdb', '_myosinhaed.pdb'), "w") as file:
        file.writelines(chain2_lines)
            
    print(f"Modified PDB file has been written to {output_file}.")

from pathlib import Path

folder_path = Path(RAW_FOLDER)

for file_path in folder_path.iterdir():
    if file_path.is_file():
        if any([f'MyoHead_{idx_of_interest}' in str(file_path) for idx_of_interest in idxs_of_interest]):
            out_file_path = str(file_path).replace('MyoHead_', 'Canonacalized_MyoHead_').replace(RAW_FOLDER, PROCESSED_FOLDER)
            print(f"Processing file: {file_path}")
            split_pdb(file_path, out_file_path)
        # Add your file processing code here


# def new_file_per_chain(input_file, output_path):
#     with open(input_file, "r") as file:
#         lines = file.readlines()

#     pdb_file_1 = []
#     pdb_file_2 = []
#     for line in lines:
#         if (
#             line.startswith("ATOM")
#             or line.startswith("HETATM")
#             or (line.startswith("TER") and len(line) > 20)
#         ):
#             try:
#                 chain_id = line[21]  # Extract current chain identifier
#             except:
#                 pass
#             # If chain id is B save to one file...
#             if chain_id == "A":
#                 pdb_file_1.append(line)
#             elif chain_id == "B":
#                 pdb_file_2.append(line)
#         else:
#             # Append lines to both files if the are just boilerplate lines
#             pdb_file_1.append(line)
#             pdb_file_2.append(line)

#     with open(f"{output_path}_chainA.pdb", "w") as file:
#         file.writelines(pdb_file_1)

#     with open(f"{output_path}_chainB.pdb", "w") as file:
#         file.writelines(pdb_file_2)


# def reindex_chain_B(input_file, output_path, increment=1000):
#     with open(input_file, "r") as file:
#         lines = file.readlines()

#     pdb_file = []
#     for line in lines:
#         if (
#             line.startswith("ATOM")
#             or line.startswith("HETATM")
#             or (line.startswith("TER") and len(line) > 20)
#         ):
#             try:
#                 chain_id = line[21]  # Extract current chain identifier
#             except:
#                 pass
#             # If chain id is B save to one file...
#             if chain_id == "A":
#                 pdb_file.append(line)
#             elif chain_id == "B":
#                 # Extract the residue number from columns 23-26 (1-based, hence 22-26 in 0-based)
#                 res_num = int(line[22:26])
#                 # Increment the residue number
#                 new_res_num = res_num + increment
#                 # Format the new residue number into the line, ensuring it stays right-aligned
#                 new_line = line[:22] + f"{new_res_num:4}" + line[26:]
#                 pdb_file.append(new_line)
#         else:
#             pdb_file.append(line)

#     with open(f"{output_path}.pdb", "w") as file:
#         file.writelines(pdb_file)

# def split_chain_B(input_file, output_path, cutoff=1000, decrement=-1000):
#     with open(input_file, "r") as file:
#         lines = file.readlines()

#     pdb_file = []
#     for line in lines:
#         if (
#             line.startswith("ATOM")
#             or line.startswith("HETATM")
#             or (line.startswith("TER") and len(line) > 20)
#         ):
#             try:
#                 chain_id = line[21]  # Extract current chain identifier
#             except:
#                 pdb_file.append(line)
#             if chain_id == "A":
#                 pdb_file.append(line)
#             elif chain_id == "B":
#                 res_num = int(line[22:26])
#                 if res_num < cutoff:
#                     pdb_file.append(line)
#                 else: 
#                     # Split all residues above the cutoff in chain B into chain C
#                     new_res_num = res_num + decrement
#                     new_line = line[:21] + "C" +  f"{new_res_num:4}" + line[26:] 
#                     pdb_file.append(new_line)
#         else:
#             pdb_file.append(line)

#     with open(f"{output_path}", "w") as file:
#         file.writelines(pdb_file)



# # for input_pdb in files:
# #     reindex_chain_B(
# #         input_pdb,
# #         f"{ROOT}/canonicalized/{input_pdb.split('/')[-1].replace('.pdb', '')}",
# #     )
# from pathlib import Path
# import os

# OLD_FOLDER = 'HADDOCK_results'
# NEW_FOLDER = 'canonicalized_HADDOCK_results'
# directory = Path(os.path.join(ROOT, OLD_FOLDER))

# # Iterate through each item in the directory
# for item in directory.iterdir():
#     if item.is_file():
#         split_chain_B(
#             str(item),
#             str(item).replace(OLD_FOLDER, NEW_FOLDER),
#         )


