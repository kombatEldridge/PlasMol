for dt in 0.01 0.005 0.001 0.0005 0.0001; do
  mkdir "$dt"
  cd "$dt"
  cp /project/bldrdge1/PlasMol/moleculeFiles/benzene-xy.in ./benzene.in
  echo "#!/bin/bash
#SBATCH --mem=50G
#SBATCH --partition=acomputeq
#SBATCH --job-name=qm
#SBATCH --time=14-00:00:00
#SBATCH --export=NONE

export QT_QPA_PLATFORM=minimal

module load meep/1.29
/opt/ohpc/pub/apps/uofm/python/3.9.13/bin/python3 /project/bldrdge1/PlasMol/qm/main.py -b benzene.in -vv -i 1e-15 -l plasmol_hpc.log -dt $dt -d x
" > submit_plasmol.sh
  cd ..
done