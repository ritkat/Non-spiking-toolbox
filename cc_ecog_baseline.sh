#!/bin/bash
#SBATCH --cpus-per-task=16
#SBATCH --time=10:00:00 # DD-HH:MM:SS
#SBATCH --mem-per-cpu=4GB
#SBATCH --array=100-200
#SBATCH --job-name=ecog_encoding

echo "Moving files"
cp -r $HOME/Non-spiking-toolbox $SLURM_TMPDIR/Non-spiking-toolbox
cd $SLURM_TMPDIR/Non-spiking-toolbox

echo "Starting application"
mkdir -p "$HOME/ecog_results_features/"

if $HOME/env/bin/python baseline_parallel.py --run $SLURM_ARRAY_TASK_ID ; then
    echo "Copying results"
    mv "accuracy_log_$SLURM_ARRAY_TASK_ID.csv" "$HOME/ecog_results_features/"
fi

wait