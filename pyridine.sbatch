#!/bin/bash
#SBATCH --mem-per-cpu   500M
#SBATCH --partition     computeq
#SBATCH --job-name      rttddft
#SBATCH --output        pyridine.out

module load python/3.9.13
export QT_QPA_PLATFORM="minimal"
/public/apps/python/3.9.13/bin/python3 /home/bldrdge1/software/bohr_dev/bohr.py pyridine.in
