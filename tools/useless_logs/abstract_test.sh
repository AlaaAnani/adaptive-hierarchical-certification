#!/bin/bash
#SBATCH -p gpu22,gpu20
#SBATCH --gres gpu:1
#SBATCH -t 1-12:00:00 

conda init
conda activate HrNet
echo $CUDA_VISIBLE_DEVICES
echo $SLURM_ARRAY_TASK_ID
export id=$SLURM_ARRAY_TASK_ID

output_file_name="abstract_slurm_test.out"

cd /BS/mlcysec2/work/robust-segmentation/
python /BS/mlcysec2/work/robust-segmentation/code/hrnet_seg/tools/abstract_slurm_test.py > /BS/mlcysec2/work/robust-segmentation/code/hrnet_seg/slurm_logs/$output_file_name