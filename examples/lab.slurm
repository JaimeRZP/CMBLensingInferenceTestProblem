#!/bin/bash
#SBATCH -A m3058
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 1:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1

export SLURM_CPU_BIND="cores"
cd /global/homes/x/xzackli/src/CMBLensingInferenceTestProblem/examples
module load cudatoolkit
module load julia/1.9.4

julia --project=. -e "import Pkg; Pkg.instantiate(); Pkg.resolve()"
julia --project=. -e "import CUDA; CUDA.set_runtime_version!(\"local\")"
srun julia --project=. Lab_CMBLens.jl
