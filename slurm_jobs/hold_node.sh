#!/bin/bash

#SBATCH --job-name=hold_node
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_jobs/output/hold_node_%j.out

# Holds a node/GPU allocation with no other purpose - use `srun --jobid=<id>
# --overlap -w <node> --pty bash` to get an interactive shell on it (see
# vscode.sh's own tunnel, which this replaces when the VS Code CLI tunnel
# itself isn't the point - just direct compute access).

sleep infinity
