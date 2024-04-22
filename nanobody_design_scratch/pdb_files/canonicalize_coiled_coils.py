import pdb

# this may only work for the ones I created...
ROOT = "/Users/yitongtseo/Documents/GitHub/ethical_necromancy/nanobody_design_scratch/pdb_files"
files = [
    f"{ROOT}/raw/2fxm.pdb",
    f"{ROOT}/raw/2fxm.pdb",
    f"{ROOT}/raw/2fxm.pdb",
    f"{ROOT}/raw/2fxm.pdb",
    f"{ROOT}/raw/2fxo_1.pdb",
    f"{ROOT}/raw/2fxo_2.pdb",
    f"{ROOT}/raw/relaxedrabbit_myosin_0to100.pdb",
    f"{ROOT}/raw/relaxedrabbit_myosin_50to150.pdb",
    f"{ROOT}/raw/relaxedrabbit_myosin_100to200.pdb",
    f"{ROOT}/raw/relaxedrabbit_myosin_150to250.pdb",
    f"{ROOT}/raw/relaxedrabbit_myosin_200to300.pdb",
    f"{ROOT}/raw/relaxedrabbit_myosin_250to350.pdb",
    f"{ROOT}/raw/relaxedrabbit_myosin_300to400.pdb",
    f"{ROOT}/raw/relaxedrabbit_myosin_350to450.pdb",
    f"{ROOT}/raw/relaxedrabbit_myosin_400to500.pdb",
    f"{ROOT}/raw/relaxedrabbit_myosin_450to550.pdb",
    f"{ROOT}/raw/relaxedrabbit_myosin_500to600.pdb",
    f"{ROOT}/raw/relaxedrabbit_myosin_550to650.pdb",
    f"{ROOT}/raw/relaxedrabbit_myosin_600to700.pdb",
    f"{ROOT}/raw/relaxedrabbit_myosin_650to750.pdb",
    f"{ROOT}/raw/relaxedrabbit_myosin_700to800.pdb",
    f"{ROOT}/raw/relaxedrabbit_myosin_750to850.pdb",
    f"{ROOT}/raw/relaxedrabbit_myosin_800to900.pdb",
    f"{ROOT}/raw/relaxedrabbit_myosin_850to950.pdb",
    f"{ROOT}/raw/relaxedrabbit_myosin_900to1000.pdb",
    f"{ROOT}/raw/relaxedrabbit_myosin_950to1050.pdb",
    f"{ROOT}/raw/relaxedrabbit_myosin_1000to1077.pdb",
]


def change_chains(input_file, output_path):
    with open(input_file, "r") as file:
        lines = file.readlines()

    pdb_file_1 = []
    pdb_file_2 = []
    for line in lines:
        if (
            line.startswith("ATOM")
            or line.startswith("HETATM")
            or (line.startswith("TER") and len(line) > 20)
        ):
            try:
                chain_id = line[21]  # Extract current chain identifier
            except:
                pass
            # If chain id is B save to one file...
            if chain_id == "A":
                pdb_file_1.append(line)
            elif chain_id == "B":
                pdb_file_2.append(line)
        else:
            # Append lines to both files if the are just boilerplate lines
            pdb_file_1.append(line)
            pdb_file_2.append(line)

    with open(f"{output_path}_chainA.pdb", "w") as file:
        file.writelines(pdb_file_1)

    with open(f"{output_path}_chainB.pdb", "w") as file:
        file.writelines(pdb_file_2)


for input_pdb in files:
    change_chains(
        input_pdb,
        f"{ROOT}/canonicalized/{input_pdb.split('/')[-1].replace('.pdb', '')}",
    )
