This is a repository made to track my progress on my Ph.D. project. The aim of the project is to produce a reliable program that simulates the affects of an organic molecule on the plasmonic response of a nanoparticle.

### Running on bigblue
```
module load meep/1.28
module load python/3.9.13/gcc.8.5.0 
alias meepy='/opt/ohpc/pub/apps/uofm/python/3.9.13/bin/python3'
# mpi4py and openmpi 4 issue needs to be covered using the next line
OMPI_MCA_btl='self,tcp,vader' 

meepy meepPlasmon.py ../run-Files/pyridine.in
```