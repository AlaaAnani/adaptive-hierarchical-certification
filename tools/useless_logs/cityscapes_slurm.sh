#!/bin/bash
#SBATCH -p gpu20
#SBATCH --gres gpu:1
conda init
conda activate HrNet
echo $CUDA_VISIBLE_DEVICES
echo $SLURM_ARRAY_TASK_ID
export id=$SLURM_ARRAY_TASK_ID

output_file_name="cityscapes_slurm.out"

cd /BS/mlcysec2/work/robust-segmentation/
python /BS/mlcysec2/work/robust-segmentation/code/hrnet_seg/tools/cityscapes_slurm.py --jobid 1 --numjobs 1 > /BS/mlcysec2/work/robust-segmentation/code/hrnet_seg/slurm_logs/$output_file_name