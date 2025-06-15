import subprocess
import os
from datetime import datetime

# Define the root directory for saving scripts and logs
root_dir = "/fs/nexus-scratch/zahirmd/act/models/"
os.makedirs(f'{root_dir}/slurm_scripts', exist_ok=True)
os.makedirs(f'{root_dir}/slurm_logs', exist_ok=True)

# Define the path to the subfolder where train_rl_mw.py is located
train_script_path = "/fs/nexus-scratch/zahirmd/act"

# Define environments and seeds
# environments = ["Assembly", "BoxClose", "CoffeePush", "StickPull", "ButtonPress", "ButtonPressTopdownWall", "Reach", "DrawerOpen", "ReachWall", "CoffeeButton"]
# environments = ["ButtonPressTopdownWall", "CoffeeButton"]
# seeds = [0, 1, 2, 3, 4]

# SLURM job parameters
partition = "scavenger"
qos = "scavenger"
account = "scavenger"
time = "4:00:00"
memory = "32gb"
gres = "gpu:1"


# for env in environments:
for i in range(0, 1):
# for seed in seeds:
    job_name = f"act_training_{i}"
    # job_dir = f"{root_dir}/{job_name}"
    # os.makedirs(job_dir, exist_ok=True)

    # Construct the command to run the Python script with arguments
    # episodes = 1000
    # save_gifs = 0
    # frame_stack = 1
    # rl_image_size = 224
    # end_on_success = True
    python_command = (
        f"python imitate_episodes.py "
        f"--task_name real_world " 
        f"--ckpt_dir {root_dir}/chkpt_real_world " 
        f"--policy_class ACT "
        f"--kl_weight 10 " 
        f"--chunk_size 100 "
        f"--hidden_dim 512 " 
        f"--batch_size 8 " 
        f"--dim_feedforward 3200 " 
        f"--num_epochs 2000 "
        f"--lr 1e-5 "
        f"--seed 0"
    )

    # Create the SLURM job script content
    job_script_content = f'''#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={root_dir}/slurm_logs/%x.%j.out
#SBATCH --time={time}
#SBATCH --partition={partition}
#SBATCH --qos={qos}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem={memory}
#SBATCH --gres={gres}
#SBATCH --account={account}

# Load necessary modules and set environment paths
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate aloha
export PYTHONPATH=$PWD:$PYTHONPATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export MUJOCO_PY_MUJOCO_PATH=~/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin

# Change directory to the folder where train_rl_mw.py is located
cd {train_script_path}

# Run the training script with the specified parameters
srun bash -c "{python_command}"
'''

    # Write the job script to a file
    job_script_path = f'{root_dir}/slurm_scripts/submit_job__{job_name}.sh'
    with open(job_script_path, 'w') as job_script_file:
        job_script_file.write(job_script_content)

    # Submit the job using sbatch
    subprocess.run(['sbatch', job_script_path])

    # Print the job submission info
    print(f'Job submitted for generating dataset: {env}')
