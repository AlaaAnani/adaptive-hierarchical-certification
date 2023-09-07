#!/bin/bash
#SBATCH -p gpu20
#SBATCH -a 1-2
#SBATCH --gres gpu:1
conda init
conda activate HrNet
echo $CUDA_VISIBLE_DEVICES
echo $SLURM_ARRAY_TASK_ID
export id=$SLURM_ARRAY_TASK_ID

output_file_name="cityscapes_slurm_$SLURM_ARRAY_TASK_ID.out"

cd /BS/mlcysec2/work/robust-segmentation/
python /BS/mlcysec2/work/robust-segmentation/code/hrnet_seg/tools/cityscapes_slurm.py --jobid $SLURM_ARRAY_TASK_ID --numjobs 10 > /BS/mlcysec2/work/robust-segmentation/code/hrnet_seg/slurm_logs/$output_file_name