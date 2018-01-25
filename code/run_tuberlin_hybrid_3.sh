#!/bin/bash
#SBATCH -A vishal.b
#SBATCH -n 1
#SBATCH --gres=gpu:3
#SBATCH --mem-per-cpu=2048
#SBATCH --time=48:00:00
#SBATCH --mincpus=24
module add cuda/8.0
module add cudnn/7-cuda-8.0

array[0]=`echo $CUDA_VISIBLE_DEVICES | cut -d"," -f1`
array[1]=`echo $CUDA_VISIBLE_DEVICES | cut -d"," -f2`
array[2]=`echo $CUDA_VISIBLE_DEVICES | cut -d"," -f3`

srun bash clean.sh; CUDA_VISIBLE_DEVICES="${array[0]}" python3 main.py --dataset='tuberlin' --data_dir='../data/' --nClasses=250 --workers=4 --epochs=400 --batch-size=128 --testbatchsize=4 --learningratescheduler='decayscheduler' --decayinterval=40 --decaylevel=2 --optimType='adam' --nesterov --tenCrop --maxlr=0.002 --minlr=0.00005 --weightDecay=0 --binaryWeight --binStart=2 --binEnd=100 --model_def='squeezenethybrid' --name='tuberlin_squeezenethybrid' | tee "textlogs/tuberlin_squeezenethybrid.txt" &
srun bash clean.sh; CUDA_VISIBLE_DEVICES="${array[1]}" python3 main.py --dataset='tuberlin' --data_dir='../data/' --nClasses=250 --workers=4 --epochs=400 --batch-size=128 --testbatchsize=4 --learningratescheduler='decayscheduler' --decayinterval=40 --decaylevel=2 --optimType='adam' --nesterov --tenCrop --maxlr=0.002 --minlr=0.00005 --weightDecay=0 --binaryWeight --binStart=2 --binEnd=100 --model_def='squeezenethybridv2' --name='tuberlin_squeezenethybridv2' | tee "textlogs/tuberlin_squeezenethybridv2.txt" &
srun bash clean.sh; CUDA_VISIBLE_DEVICES="${array[2]}" python3 main.py --dataset='tuberlin' --data_dir='../data/' --nClasses=250 --workers=4 --epochs=400 --batch-size=128 --testbatchsize=4 --learningratescheduler='decayscheduler' --decayinterval=40 --decaylevel=2 --optimType='adam' --nesterov --tenCrop --maxlr=0.002 --minlr=0.00005 --weightDecay=0 --binaryWeight --binStart=2 --binEnd=100 --model_def='squeezenethybridv3' --name='tuberlin_squeezenethybridv3' | tee "textlogs/tuberlin_squeezenethybridv3.txt"
