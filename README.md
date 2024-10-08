## To Do List:
1. The time step of bohr and of Meep are incongruent about two orders of magnitude. As a result, the field that is reported from bohr is about 100 bohr time steps old by the time it is considered by Meep. I suspect we will have to average some things in order for this to work which is not optimal. 
    a. The time step in Meep is dependent on the resolution of the simulation, so I suppose I could increase the resolution 100 fold and see if the the math says the time steps work out. However, we all know increasing resolution will cost computationally. Maybe BigBlue can handle it?
    b. Many other things are dependent on resolution and I suspect that even if BigBlue were able to handle it, there would be numeric instabilities involved (citation needed).
2. As we gain experience with Meep, I would like to investigate similar structures as I did in DDSCAT. Specifically, I would love to see the real-time propagation of the field in the Au@SiO2@Au multilayer and the Au-Ag systems. 


## Introduction
This is a repository made to track my progress on my Ph.D. project. The aim of the project is to produce a reliable program that simulates the affects of an organic molecule on the plasmonic response of a nanoparticle.

## Getting Started 
### Required Packages
The base of this project runs on `python3` (specifically tested on 3.9.18). Almost every computer system will have this installed. Other required packages include bohr and Meep.

### Installing Meep
Instructions on how to install MEEP can be found on their docs [here](https://meep.readthedocs.io/en/master/Installation/). Do keep in mind whether your installation will be on an HPC/cluster or locally on a singular machine.

### Installing bohr
bohr is currently a private build and details on usage remain TBD.

## Running

### Running locally
Once you establish a Meep conda environment, you'll need to activate it.


../bohr_dev/plasmol2.py -m meep.in -b pyridine.in -l plasmol2.out -vv

### Running on an HPC/Cluster
Specifically on the BigBlue at the University of Memphis:
```bash
module load meep/1.28

module list
# Currently Loaded Modules:
# 1) fftw/3.3.10/gcc-8.5.0/openmpi-4.1.6
# 2) openblas/0.3.26/gcc-8.5.0/nonthreaded   
# 3) lapack/3.12.0                           
# 4) python/3.9.13/gcc.8.5.0               
# 5) hdf5/1.14.3/gcc-8.5.0/openmpi-4.1.6   
# 6) openmpi/4.1.6/gcc.8.5.0/noncuda
# 7) gsl/2.7.1/gcc-8.5.0
# 8) meep/1.28

# mpi4py and openmpi 4 issue needs to be covered using the next line
OMPI_MCA_btl='self,tcp,vader' 

# To execute from /molecule-Files:
python3 ../bohr_dev/plasmol.py pyridine.in
```



# Questions for Nascimento

### Out of plane response
When we give Bohr the following E field measurements at the three time steps, 
|   | $t_n$                   | $t_n + dt/2$           | $t_{n+1}$               |
|---|-------------------------|------------------------|-------------------------|
| x | 0.0                     | 0.0                    | 0.0                     |
| y | -2.9590431971790376e-31 | -1.183617278871615e-30 | -1.183617278871615e-30  |
| z | -3.2321144001108524e-13 | -6.362250103160091e-13 | -1.2372383136685856e-12 |

it returns the following array of its response in the x, y, and z direction
|   | $t_{n+1}$               |
|---|-------------------------|
| x | 1.4210854715202004e-14  |
| y | -8.837375276016246e-14  |
| z | -7.776588659828583e-14  |

You can clearly see that the incoming electric field is excited in the z direction, but the output of the molecule is excited in all three directions. *Is this a realistic result?*

### How to measure the incoming E field
Currently, the electric field is measured at the specific voxel (term for 3D pixel) of the molecule. When the response is taken from Bohr, it outputs at that very same voxel. That means at the second step, we measure both the field coming in and also the response field and feed that back as the input field. *What is an appropriate way to measure the incoming field?*

These fields have directions, and we can measure the flux of the electric fields in and out of the molecule's voxel.

![img](FluxBox.jpg)
