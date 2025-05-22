import os
import numpy as np
import bohr_internals.constants as constants

# I replaced all prints with blanks to not clog up runtime messages.

def read_input(inputfile,options):
    #read input file and separates it into three classes: molecule, method, and basis
    molecule = {}
    method = {}
    basis = {} 
    
    options.inputfile = inputfile
    #set default parameters
    method["e_conv"] = 1e-6 
    method["d_conv"] = 1e-6
    method["maxiter"] = 500 
    method["diis"] = True 
    method["zora"] = False 
    method["so_correction"] = False 
    method["so_scf"] = False 
    method["in_core"] = False 
    method["reference"] = "restricted" 
    method["occupied"] = []
    method["nroots"] = 5
    method["do_cis"] = False  
    method["do_tdhf"] = False  
    method["do_cpp"] = False  
    method["resplimit"] = 1e-20
    method["propagator"] = "rk4"

    molecule["charge"] = 0
    molecule["spin"] = 0
    molecule["units"] = "angstroms"
    molecule["nocom"] = False

    basis["n_radial"]  = 49
    basis["n_angular"] = 35


    read_mol = False
    count = 0
    atoms = []
    coords = {}
    f = open(inputfile,"r")
    for line in f:
        if "#" in line:
           continue
        if "end molecule" in line:
            read_mol = False
    
        if (len(line.split()) > 1) and (read_mol is True):
            atoms.append(line.split()[0])
            coords[str(atoms[count])+str(count+1)] = \
            np.array((float(line.split()[1]),float(line.split()[2]),\
            float(line.split()[3])))
            count += 1
    
        if "start molecule" in line:
            read_mol = True
            count = 0
    
        if "basis" in line:
            if "library" in line:
              basis["name"] = os.path.abspath(os.path.dirname(__file__))+"/basis/"+str(line.split()[2])
            else: 
              basis["name"] = str(line.split()[1])

        if "n_radial" in line:
            basis["n_radial"] = int(line.split()[1])

        if "n_angular" in line:
            basis["n_angular"] = int(line.split()[1])
    
        if "method" in line:
            method["name"] = str(line.split()[1])
            options.method = str(line.split()[1])
    
        if "reference" in line:
            method["reference"] = str(line.split()[1])

        if "xc" in line:
            if len(line.split()) == 3: #read exchange and correlation separately
              method["xc"] = str(line.split()[1])+","+str(line.split()[2])
              options.xc = str(line.split()[1])+","+str(line.split()[2])
            elif len(line.split()) == 2: #read alias to xc functional
              options.xc = str(line.split()[1]) #+","
        if "e_conv" in line:
            method["e_conv"] = float(line.split()[1])
            options.e_conv   = float(line.split()[1])

        if "d_conv" in line:
            method["d_conv"] = float(line.split()[1])
            options.d_conv   = float(line.split()[1])

        if "ft_gamma" in line:
            options.gamma   = float(line.split()[1])

        if "maxiter" in line:
            method["maxiter"] = int(line.split()[1])
            options.maxiter   = int(line.split()[1])

        if "nroots" in line:
            method["nroots"] = int(line.split()[1])
            options.nroots = int(line.split()[1])
        if "grid_level" in line:
            options.grid_level = int(line.split()[1])
        
        if "batch_size" in line:
            options.batch_size = int(line.split()[1])

        if "guess_mos" in line:
            options.guess_mos = str(line.split()[1])
            options.guess_mos_provided = True
        if "cosmo" in line:
            if len(line.split()) > 1:
              options.cosmo_epsilon = float(line.split()[1])
            options.cosmo = True

            

        if "occupied" in line:
            method["occupied"] = [int(line.split()[1]), int(line.split()[2])]
            options.occupied = [int(line.split()[1]), int(line.split()[2])]
            options.cvs = True

        if "virtual" in line:
            options.virtual = [int(line.split()[1]), int(line.split()[2])]
            options.reduced_virtual = True

        if "couple_states" in line:
            options.occupied1 = [int(line.split()[1]), int(line.split()[2])]
            options.occupied2 = [int(line.split()[3]), int(line.split()[4])]
            options.couple_states = True

        if "frequencies" in line:
            if "eV" in line:
              options.frequencies = np.asarray([float(line.split()[1]), float(line.split()[2]), float(line.split()[3]), float(line.split()[4])])*constants.ev2au
            else: 
              options.frequencies = np.asarray([float(line.split()[1]), float(line.split()[2]), float(line.split()[3]), float(line.split()[4])])

        if "so_scf" in line:
            method["so_scf"] = True
            options.so_scf = True

        if "akonly" in line:
            options.akonly = True
            options.nofxc = True
        if "fonly" in line:
            options.fonly = True
            options.nofxc = True
        if "jkonly" in line:
            options.jkonly = True
            options.nofxc = True
        if "jonly" in line:
            options.jonly = True
            options.nofxc = True
        if "nofxc" in line:
            options.nofxc = True

        if "cartesian" in line:
            options.cartesian = True
        if "spherical" in line:
            options.cartesian = False
        if "tddft_plus_tb" in line:
            options.plus_tb = True

        if "noscf" in line:
            method["noscf"] = True
            options.noscf = True

        if "tdscf_in_core" in line:
            options.tdscf_in_core = True

        if "direct_diagonalization" in line:
            options.direct_diagonalization = True

        if "in_core" in line:
            method["in_core"] = True

        if "do_tda" in line:
            method["do_tda"] = True
            options.do_tda = True

        if "do_cis" in line:
            method["do_cis"] = True
            options.do_cis = True

        if "do_tdhf" in line:
            method["do_tdhf"] = True
            options.do_tdhf = True

        if "do_cpp" in line:
            method["do_cpp"] = True
            options.do_cpp   = True

        if "charge" in line:
            molecule["charge"] = float(line.split()[1])
            options.charge = float(line.split()[1])

        if "spin" in line:
            molecule["spin"] = int(line.split()[1]) 
            options.spin = int(line.split()[1])

        if "mult" in line:
            options.mult = int(line.split()[1])
            molecule["spin"] = int(line.split()[1])-1 
            options.spin = int(line.split()[1])-1 
            

        if "units" in line:
            molecule["units"] = str(line.split()[1])
        
        if "nocom" in line:
            molecule["nocom"] = True

        if "sflip" in line:
            options.spin_flip = True

        if "swap_mos" in line:
           orig_str = line.split(",")[0].split()[1:]
           swap_str = line.split(",")[1].split()
           if len(orig_str) != len(swap_str):
             exit("Incorrect dimensions for orbital swap!")
           orig = [int(i)-1 for i in orig_str]
           swap = [int(i)-1 for i in swap_str]
  
           options.swap_mos = [orig,swap]

        if "diis" in line:
            if line.split()[1] == "true":
                method["diis"] = True
                options.diis   = True
            elif line.split()[1] == "false":
                method["diis"] = False
                options.diis   = False

        if "relativistic" in line:
            options.relativistic = line.split()[1]

        if "memory" in line:
            options.memory = float(line.split()[1])

        if "B_field_amplitude" in line:
            options.B_field_amplitude = float(line.split()[1])
        if "E_field_amplitude" in line:
            options.E_field_amplitude = float(line.split()[1])
        if "B_field_polarization" in line:
            options.B_field_polarization = int(line.split()[1])
        if "E_field_polarization" in line:
            options.E_field_polarization = int(line.split()[1])

        if "so_correction" in line:
            if line.split()[1] == "true":
                method["so_correction"] = True 

        if "roots_lookup_table" in line:
            low = int(line.split()[1])
            high = int(line.split()[2])+1
            options.roots_lookup_table = np.arange(low,high,1)

        if "resplimit" in line:
            method["resplimit"] = float(line.split()[1])

        if "propagator" in line:
            method["propagator"] = str(line.split()[1])

    molecule["atoms"] = atoms
    molecule["coords"] = get_coords(atoms,coords,molecule)

    #check/get multiplicity
    molecule["n_electrons"] = get_nelectrons(molecule["atoms"]) - molecule["charge"]
    molecule["e_nuclear"] = get_enuc(molecule)
    molecule["tb_gamma"] = get_tbgamma_w(molecule)[0]
    molecule["tb_w"] = get_tbgamma_w(molecule)[1]

    print_molecule_info(molecule)
    print_method_info(method)
    print_basis_info(basis)

    return molecule, method, basis

def get_coords(atoms,coords,mol):
    if (mol["units"] == "angstroms"):
        for Ai, A in enumerate(atoms):
            coords[A+str(Ai+1)] *= constants.angs2bohr

    #center of mass
    mass = np.zeros((len(atoms)))
    com  = np.zeros((3))
    for Ai, A in enumerate(atoms):
        xyz = coords[A+str(Ai+1)]
        mass[Ai] = constants.masses[A.upper()]
        com += xyz * mass[Ai] 

    mol["com"] = com
        
    if (mol["nocom"] is False):    
        for Ai, A in enumerate(atoms):
            coords[A+str(Ai+1)] -= com/np.sum(mass) 

     
    return coords    


def get_enuc(mol):
    atoms = mol["atoms"]
    coords = mol["coords"]
    natoms = len(atoms)
    enuc = 0.
    for a in range(natoms):
        Za = constants.Z[atoms[a].upper()]
        atom = atoms[a]+str(a+1)
        RA = np.array([coords[atom][0],coords[atom][1],coords[atom][2]])
        for b in range(0,a):
            Zb = constants.Z[atoms[b].upper()]
            atom = atoms[b]+str(b+1)
            RB = np.array([coords[atom][0],coords[atom][1],coords[atom][2]])
            RAB = np.linalg.norm(RA-RB)
            enuc += Za*Zb/RAB
    return enuc

def get_tbgamma_w(mol):
    atoms = mol["atoms"]
    coords = mol["coords"]
    natoms = len(atoms)
    enuc = 0.
    tbgamma = np.zeros((len(atoms),len(atoms)))
    tbw = np.zeros((len(atoms),len(atoms)))
    for a in range(natoms):
        try:
          eta_a = constants.eta[atoms[a].upper()]/27.21138 
        except:
          eta_a = 0.  
        try: 
          tbw_a = constants.tbw[atoms[a].upper()] 
        except:
          tbw_a = 0.  
        tbw[a][a] = tbw_a
        atom = atoms[a]+str(a+1)
        RA = np.array([coords[atom][0],coords[atom][1],coords[atom][2]])
        for b in range(0,a):
            try:
              eta_b = constants.eta[atoms[b].upper()]/27.21138
            except:
              eta_b = 0. 
            atom = atoms[b]+str(b+1)
            RB = np.array([coords[atom][0],coords[atom][1],coords[atom][2]])
            RAB = np.linalg.norm(RA-RB)
            if RAB == 0:
              S = 5.* eta_a/16. 
            elif (RAB != 0) and abs(eta_a-eta_b) < 1e-12:
              S = np.exp(-eta_a*RAB) * (48. + 33.*eta_a*RAB + 9.*eta_a**2 * RAB**2 + eta_a**3 * RAB**3)/(48.*RAB)
            else:
              S = np.exp(-eta_a*RAB)*(eta_b**4 * eta_a/(2.*(eta_a**2-eta_b**2)**2) - (eta_b**6 - 3.*eta_b**4 * eta_a**2)/((eta_a**2 - eta_b**2)**3 *RAB))
              S += np.exp(-eta_b*RAB)*(eta_a**4 * eta_b/(2.*(eta_b**2-eta_a**2)**2) - (eta_a**6 - 3.*eta_a**4 * eta_b**2)/((eta_b**2 - eta_a**2)**3 *RAB))
            tbgamma[a][b] += 1./RAB - S
    return tbgamma, tbw

def get_nelectrons(atoms):
    nel = 0
    for atom in atoms: 
        nel += constants.Z[atom.upper()]
    return nel

def print_molecule_info(mol):
    atoms = mol["atoms"]
    coords = mol["coords"]
    nel = mol["n_electrons"]
    mult = mol["spin"]+1
    charge = mol["charge"]
    natoms = len(atoms)
    print("", end ="") # print("    Molecule Info")
    print("", end ="") # print("    -------------")
    print("", end ="") # print("")
    print("", end ="") # print("    Number of atoms    : %i"%(natoms))
    print("", end ="") # print("    Number of electrons: %i"%(nel))
    print("", end ="") # print("    Charge             : %i"%(charge))
    print("", end ="") # print("    Multiplicity       : %i"%(mult))
    print("", end ="") # print("    Geometry [a0]:")
    for a in range(natoms):
        atom = atoms[a]+str(a+1)
        print("", end ="") # print("    %5s %20.12f %20.12f %20.12f "%(atoms[a],coords[atom][0],coords[atom][1],coords[atom][2]))
    print("", end ="") # print("")
    return None

def print_method_info(method):
    print("", end ="") # print("    Method: %s "%method["name"])
    print("", end ="") # print("    E_conv: %e "%method["e_conv"])
    print("", end ="") # print("    D_conv: %e "%method["d_conv"])
    print("", end ="") # print("    Maxiter: %i "%method["maxiter"])
    print("", end ="") # print("")
    return None
def print_basis_info(basis):
    print("", end ="") # print("    Basis set: %s "%basis["name"])
    return None
