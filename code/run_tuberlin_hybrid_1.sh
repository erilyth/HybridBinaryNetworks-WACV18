#!/bin/bash
#SBATCH -A research
#SBATCH -n 1
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2048
#SBATCH --time=48:00:00
#SBATCH --mincpus=32
module add cuda/8.0
module add cudnn/7-cuda-8.0

array[0]=`echo $CUDA_VISIBLE_DEVICES | cut -d"," -f1`
array[1]=`echo $CUDA_VISIBLE_DEVICES | cut -d"," -f2`
array[2]=`echo $CUDA_VISIBLE_DEVICES | cut -d"," -f3`

srun bash clean.sh; CUDA_VISIBLE_DEVICES="${array[0]}" python3 main.py --dataset='tuberlin' --data_dir='../data/' --nClasses=250 --workers=4 --epochs=600 --batch-size=256 --testbatchsize=16 --learningratescheduler='decayscheduler' --decayinterval=50 --decaylevel=2 --optimType='adam' --nesterov --tenCrop --maxlr=0.002 --minlr=0.00005 --weightDecay=0 --binaryWeight --binStart=2 --binEnd=7 --model_def='sketchanethybrid' --inpsize=225 --name='tuberlin_sketchanethybrid' | tee "textlogs/tuberlin_sketchanethybrid.txt" &
srun bash clean.sh; CUDA_VISIBLE_DEVICES="${array[1]}" python3 main.py --dataset='tuberlin' --data_dir='../data/' --nClasses=250 --workers=4 --epochs=600 --batch-size=256 --testbatchsize=16 --learningratescheduler='decayscheduler' --decayinterval=50 --decaylevel=2 --optimType='adam' --nesterov --tenCrop --maxlr=0.002 --minlr=0.00005 --weightDecay=0 --binaryWeight --binStart=2 --binEnd=7 --model_def='sketchanethybridv2' --inpsize=225 --name='tuberlin_sketchanethybridv2' | tee "textlogs/tuberlin_sketchanethybridv2.txt" &
srun bash clean.sh; CUDA_VISIBLE_DEVICES="${array[2]}" python3 main.py --dataset='tuberlin' --data_dir='../data/' --nClasses=250 --workers=4 --epochs=600 --batch-size=256 --testbatchsize=16 --learningratescheduler='decayscheduler' --decayinterval=50 --decaylevel=2 --optimType='adam' --nesterov --tenCrop --maxlr=0.002 --minlr=0.00005 --weightDecay=0 --binaryWeight --binStart=2 --binEnd=8 --model_def='sketchanethybrid' --inpsize=225 --name='tuberlin_sketchanethybrid_binend8' | tee "textlogs/tuberlin_sketchanethybrid_binend8.txt" &
srun bash clean.sh; CUDA_VISIBLE_DEVICES="${array[3]}" python3 main.py --dataset='tuberlin' --data_dir='../data/' --nClasses=250 --workers=4 --epochs=600 --batch-size=256 --testbatchsize=16 --learningratescheduler='decayscheduler' --decayinterval=50 --decaylevel=2 --optimType='adam' --nesterov --tenCrop --maxlr=0.002 --minlr=0.00005 --weightDecay=0 --binaryWeight --binStart=2 --binEnd=8 --model_def='sketchanethybridv2' --inpsize=225 --name='tuberlin_sketchanethybridv2_binend8' | tee "textlogs/tuberlin_sketchanethybridv2_binend8.txt"
