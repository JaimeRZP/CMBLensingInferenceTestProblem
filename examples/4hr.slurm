#!/bin/bash
#SBATCH -A m3058
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 4:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1

export SLURM_CPU_BIND="cores"
cd /global/homes/x/xzackli/src/CMBLensingInferenceTestProblem/examples
module load cudatoolkit
module load julia/1.9.4


echo "Running job with script: $1"

julia --project=. -e "import Pkg; Pkg.instantiate(); Pkg.resolve()"
julia --project=. -e "import CUDA; CUDA.set_runtime_version!(\"local\")"

echo $1 $2 $3 $4 $5 $6
srun julia --project=. $1 $2 $3 $4 $5 $6
