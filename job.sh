#!/usr/bin/env bash
#SBATCH --partition=staff-amlrt
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=3:00:00
#SBATCH --job-name=project-dev
#SBATCH --output=logs/%x__%j.log
#SBATCH --error=logs/%x__%j.log

# Prepare the environment
set -e
module --quiet load python/3.10
source ./.venv/bin/activate

astfnet-train --config /home/mila/g/ge.li/ASTF-net/config/config-transformer.yaml
