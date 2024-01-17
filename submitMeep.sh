#!/usr/bin/sh
#SBATCH --ntasks=8
#SBATCH --mem-per-cpu=5G
#SBATCH --job-name=meep_mpi
#SBATCH --partition=computeq
#SBATCH --out=meep_mpi_%J.out
#SBATCH --export=NONE

module load meep/1.28-p h5utils/1.13.2
module load python/3.9.13

mpirun /cm/shared/public/apps/python/3.9.13/bin/python3 meepAuSphere.py
