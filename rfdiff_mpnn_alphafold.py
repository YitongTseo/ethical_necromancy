def run_the_whole_shebang(
    name = "turkey_myh2_118to127", # RFDiff params
    contigs = "A:50-100",
    pdb = "6XE9",
    iterations=50,
    hotspot="A118,A119,A123,A124,A125,A126,A127",
    num_designs=4,
    chains = "A",
    num_seqs = 64, # Protein MPNN params
    initial_guess = True,
    num_recycles = 3,
    version='v1',
):
    name = name + version
    # #@title run **RFdiffusion** to generate a backbone
    # contigs = "A:50-100" #@param {type:"string"}
    # pdb = "6XE9" #@param {type:"string"}
    # iterations = 50 #@param ["25", "50", "100", "150", "200"] {type:"raw"}
    # hotspot = "A118,A119,A123,A124,A125,A126,A127" #@param {type:"string"}
    # num_designs = 4 #@param ["1", "2", "4", "8", "16", "32"] {type:"raw"}
    visual = "image" #@param ["none", "image", "interactive"]
    #@markdown ---
    #@markdown **symmetry** settings
    #@markdown ---
    symmetry = "none" #@param ["none", "auto", "cyclic", "dihedral"]
    order = 1 #@param ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"] {type:"raw"}
    add_potential = True #@param {type:"boolean"}
    #@markdown - `symmetry='auto'` enables automatic symmetry dectection with [AnAnaS](https://team.inria.fr/nano-d/software/ananas/).
    #@markdown - `chains="A,B"` filter PDB input to these chains (may help auto-symm detector)
    #@markdown - `add_potential` to discourage clashes between chains

    # determine where to save
    path = name
    while os.path.exists(f"outputs/{path}_0.pdb"):
    path = name + "_" + ''.join(random.choices(string.ascii_lowercase + string.digits, k=5))

    flags = {"contigs":contigs,
            "pdb":pdb,
            "order":order,
            "iterations":iterations,
            "symmetry":symmetry,
            "hotspot":hotspot,
            "path":path,
            "chains":chains,
            "add_potential":add_potential,
            "num_designs":num_designs,
            "visual":visual}

    for k,v in flags.items():
    if isinstance(v,str):
        flags[k] = v.replace("'","").replace('"','')

    contigs, copies = run_diffusion(**flags)

    #@title Display 3D structure {run: "auto"}
    animate = "none" #@param ["none", "movie", "interactive"]
    color = "chain" #@param ["rainbow", "chain", "plddt"]
    denoise = True
    dpi = 100 #@param ["100", "200", "400"] {type:"raw"}
    from colabdesign.shared.plot import pymol_color_list
    from colabdesign.rf.utils import get_ca, get_Ls, make_animation
    from string import ascii_uppercase,ascii_lowercase
    alphabet_list = list(ascii_uppercase+ascii_lowercase)

    def plot_pdb(num=0):
    if denoise:
        pdb_traj = f"outputs/traj/{path}_{num}_pX0_traj.pdb"
    else:
        pdb_traj = f"outputs/traj/{path}_{num}_Xt-1_traj.pdb"
    if animate in ["none","interactive"]:
        hbondCutoff = 4.0
        view = py3Dmol.view(js='https://3dmol.org/build/3Dmol.js')
        if animate == "interactive":
        pdb_str = open(pdb_traj,'r').read()
        view.addModelsAsFrames(pdb_str,'pdb',{'hbondCutoff':hbondCutoff})
        else:
        pdb = f"outputs/{path}_{num}.pdb"
        pdb_str = open(pdb,'r').read()
        view.addModel(pdb_str,'pdb',{'hbondCutoff':hbondCutoff})
        if color == "rainbow":
        view.setStyle({'cartoon': {'color':'spectrum'}})
        elif color == "chain":
        for n,chain,c in zip(range(len(contigs)),
                                alphabet_list,
                                pymol_color_list):
            view.setStyle({'chain':chain},{'cartoon': {'color':c}})
        else:
        view.setStyle({'cartoon': {'colorscheme': {'prop':'b','gradient': 'roygb','min':0.5,'max':0.9}}})
        view.zoomTo()
        if animate == "interactive":
        view.animate({'loop': 'backAndForth'})
        view.show()
    else:
        Ls = get_Ls(contigs)
        xyz, bfact = get_ca(pdb_traj, get_bfact=True)
        xyz = xyz.reshape((-1,sum(Ls),3))[::-1]
        bfact = bfact.reshape((-1,sum(Ls)))[::-1]
        if color == "chain":
        display(HTML(make_animation(xyz, Ls=Ls, dpi=dpi, ref=-1)))
        elif color == "rainbow":
        display(HTML(make_animation(xyz, dpi=dpi, ref=-1)))
        else:
        display(HTML(make_animation(xyz, plddt=bfact*100, dpi=dpi, ref=-1)))


    if num_designs > 1:
    output = widgets.Output()
    def on_change(change):
        if change['name'] == 'value':
        with output:
            output.clear_output(wait=True)
            plot_pdb(change['new'])
    dropdown = widgets.Dropdown(
        options=[(f'{k}',k) for k in range(num_designs)],
        value=0, description='design:',
    )
    dropdown.observe(on_change)
    display(widgets.VBox([dropdown, output]))
    with output:
        plot_pdb(dropdown.value)
    else:
    plot_pdb()



    rm_aa = "C" #@param {type:"string"}
    mpnn_sampling_temp = 0.1 #@param ["0.0001", "0.1", "0.15", "0.2", "0.25", "0.3", "0.5", "1.0"] {type:"raw"}
    #@markdown - for **binder** design, we recommend `initial_guess=True num_recycles=3`

    if not os.path.isfile("params/done.txt"):
    print("downloading AlphaFold params...")
    while not os.path.isfile("params/done.txt"):
        time.sleep(5)

    contigs_str = ":".join(contigs)
    opts = [f"--pdb=outputs/{path}_0.pdb",
            f"--loc=outputs/{path}",
            f"--contig={contigs_str}",
            f"--copies={copies}",
            f"--num_seqs={num_seqs}",
            f"--num_recycles={num_recycles}",
            f"--rm_aa={rm_aa}",
            f"--mpnn_sampling_temp={mpnn_sampling_temp}",
            f"--num_designs={num_designs}"]
    if initial_guess: opts.append("--initial_guess")
    if use_multimer: opts.append("--use_multimer")
    opts = ' '.join(opts)
    !python colabdesign/rf/designability_test.py {opts}


    import py3Dmol
    def plot_pdb(num = "best"):
    if num == "best":
        with open(f"outputs/{path}/best.pdb","r") as f:
        # REMARK 001 design {m} N {n} RMSD {rmsd}
        info = f.readline().strip('\n').split()
        num = info[3]
    hbondCutoff = 4.0
    view = py3Dmol.view(js='https://3dmol.org/build/3Dmol.js')
    pdb_str = open(f"outputs/{path}_{num}.pdb",'r').read()
    view.addModel(pdb_str,'pdb',{'hbondCutoff':hbondCutoff})
    pdb_str = open(f"outputs/{path}/best_design{num}.pdb",'r').read()
    view.addModel(pdb_str,'pdb',{'hbondCutoff':hbondCutoff})

    view.setStyle({"model":0},{'cartoon':{}}) #: {'colorscheme': {'prop':'b','gradient': 'roygb','min':0,'max':100}}})
    view.setStyle({"model":1},{'cartoon':{'colorscheme': {'prop':'b','gradient': 'roygb','min':0,'max':100}}})
    view.zoomTo()
    view.show()

    if num_designs > 1:
    def on_change(change):
        if change['name'] == 'value':
        with output:
            output.clear_output(wait=True)
            plot_pdb(change['new'])
    dropdown = widgets.Dropdown(
        options=["best"] + [str(k) for k in range(num_designs)],
        value="best",
        description='design:',
    )
    dropdown.observe(on_change)
    output = widgets.Output()
    display(widgets.VBox([dropdown, output]))
    with output:
        plot_pdb(dropdown.value)
    else:
    plot_pdb()

    !zip -r {path}.result.zip outputs/{path}* outputs/traj/{path}*
    files.download(f"{path}.result.zip")