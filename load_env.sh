module purge
module load cuda/10.1
module load NCCL/2.7.6-1-cuda.10.1
module load cudnn/v7.6.5.32-cuda.10.1
module load anaconda3/2020.11  # Might need to change this to change Python version
source ~/.venv/fairscale_venv/bin/activate

export MYPYPATH=~/fairscale_vinayak/stubs