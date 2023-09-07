#!/bin/bash
#SBATCH -p gpu20
#SBATCH --gres gpu:2
conda init
conda activate HrNet
echo $CUDA_VISIBLE_DEVICES
echo $SLURM_ARRAY_TASK_ID
export id=$SLURM_ARRAY_TASK_ID

output_file_name="array_job_$SLURM_ARRAY_TASK_ID.out"

cd /BS/mlcysec2/work/robust-segmentation/
python /BS/mlcysec2/work/robust-segmentation/code/hrnet_seg/tools/cityscapes_smoothing.py > /BS/mlcysec2/work/robust-segmentation/code/hrnet_seg/slurm_logs/calibrate.out