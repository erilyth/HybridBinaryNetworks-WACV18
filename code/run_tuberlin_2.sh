#!/bin/bash
#SBATCH -A research
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

srun bash clean.sh; CUDA_VISIBLE_DEVICES="${array[0]}" python3 main.py --dataset='tuberlin' --data_dir='../data/' --nClasses=250 --workers=4 --epochs=400 --batch-size=256 --testbatchsize=16 --learningratescheduler='decayscheduler' --decayinterval=40 --decaylevel=2 --optimType='adam' --nesterov --tenCrop --maxlr=0.002 --minlr=0.00005 --weightDecay=0 --model_def='resnet18' --name='tuberlin_resnet18' | tee "textlogs/tuberlin_resnet18.txt" &
srun bash clean.sh; CUDA_VISIBLE_DEVICES="${array[1]}" python3 main.py --dataset='tuberlin' --data_dir='../data/' --nClasses=250 --workers=4 --epochs=400 --batch-size=160 --testbatchsize=16 --learningratescheduler='decayscheduler' --decayinterval=40 --decaylevel=2 --optimType='adam' --nesterov --tenCrop --maxlr=0.002 --minlr=0.00005 --weightDecay=0 --binaryWeight --binStart=2 --binEnd=56 --model_def='resnetwbin18' --name='tuberlin_resnetwbin18' | tee "textlogs/tuberlin_resnetwbin18.txt" &
srun bash clean.sh; CUDA_VISIBLE_DEVICES="${array[2]}" python3 main.py --dataset='tuberlin' --data_dir='../data/' --nClasses=250 --workers=4 --epochs=400 --batch-size=160 --testbatchsize=16 --learningratescheduler='decayscheduler' --decayinterval=40 --decaylevel=2 --optimType='adam' --nesterov --tenCrop --maxlr=0.002 --minlr=0.00005 --weightDecay=0 --binaryWeight --binStart=2 --binEnd=56 --model_def='resnetfbin18' --name='tuberlin_resnetfbin18' | tee "textlogs/tuberlin_resnetfbin18.txt"
