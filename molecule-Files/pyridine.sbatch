#!/bin/bash
#SBATCH --ntasks=8
#SBATCH --mem-per-cpu=5G
#SBATCH --partition=computeq
#SBATCH --job-name=plasmonMol
#SBATCH --output=plasmonMol_%J.out
#SBATCH --export=NONE
export QT_QPA_PLATFORM="minimal"

module load meep/1.28-p h5utils/1.13.2
mpirun /cm/shared/public/apps/python/3.9.13/bin/python3 meepAuSphere.py
/public/apps/python/3.9.13/bin/python3 /home/bldrdge1/software/bohr_dev/bohr.py pyridine.in
