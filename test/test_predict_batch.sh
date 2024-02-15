#!/bin/bash

#SBATCH --job-name=generate-dpl   # job name
#SBATCH --output=dpl-inference-%N.out       # output log file
#SBATCH --time=04:00:00       # 1 hour of wall time
#SBATCH --nodes=1             # 1 GPU node
#SBATCH --ntasks-per-node=10   
#SBATCH --partition=gpu      
#SBATCH --account=pi-roux
#SBATCH --gres=gpu:1          # Request 1 GPU per node


source ~/.bashrc
conda activate /scratch/midway3/ndn/dpl


save_dir="/scratch/beagle3/ndn/DiffusionProteinLigand/inference/example_beta_encoding/${SLURM_JOB_ID}"

python -m test.predict_batch_seq \
    --ckpt_path "/scratch/beagle3/ndn/DiffusionProteinLigand/beta_train/retrain/10181134/lightning_logs/version_10181134/checkpoints/epoch=328-val_loss=1.20.ckpt"\
    --output_dir "$save_dir" \
    --fasta "./test/test_cases.fasta" \
    --num_samples 2 \
    --num_steps 1000