#!/bin/bash
#SBATCH --mem=50G
#SBATCH --partition=acomputeq
#SBATCH --job-name=plasmol
#SBATCH --export=NONE

export QT_QPA_PLATFORM="minimal"

module load meep/1.29
/opt/ohpc/pub/apps/uofm/python/3.9.13/bin/python3 /project/bldrdge1/PlasMol/bohr/driver.py -m meep.in -b pyridine.in -vv -l plasmol_hpc.log

