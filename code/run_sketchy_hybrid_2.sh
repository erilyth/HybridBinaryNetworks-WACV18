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

srun bash clean.sh; CUDA_VISIBLE_DEVICES="${array[0]}" python3 main.py --dataset='sketchyrecognition' --data_dir='../data/' --nClasses=125 --workers=4 --epochs=400 --batch-size=256 --testbatchsize=16 --learningratescheduler='decayscheduler' --decayinterval=40 --decaylevel=2 --optimType='adam' --nesterov --tenCrop --maxlr=0.002 --minlr=0.00005 --weightDecay=0 --binaryWeight --binStart=2 --binEnd=100 --model_def='resnethybrid18' --name='sketchy_resnethybrid18' | tee "textlogs/sketchy_resnethybrid18.txt" &
srun bash clean.sh; CUDA_VISIBLE_DEVICES="${array[1]}" python3 main.py --dataset='sketchyrecognition' --data_dir='../data/' --nClasses=125 --workers=4 --epochs=400 --batch-size=160 --testbatchsize=16 --learningratescheduler='decayscheduler' --decayinterval=40 --decaylevel=2 --optimType='adam' --nesterov --tenCrop --maxlr=0.002 --minlr=0.00005 --weightDecay=0 --binaryWeight --binStart=2 --binEnd=100 --model_def='resnethybridv218' --name='sketchy_resnethybridv218' | tee "textlogs/sketchy_resnethybridv218.txt" &
srun bash clean.sh; CUDA_VISIBLE_DEVICES="${array[2]}" python3 main.py --dataset='sketchyrecognition' --data_dir='../data/' --nClasses=125 --workers=4 --epochs=400 --batch-size=160 --testbatchsize=16 --learningratescheduler='decayscheduler' --decayinterval=40 --decaylevel=2 --optimType='adam' --nesterov --tenCrop --maxlr=0.002 --minlr=0.00005 --weightDecay=0 --binaryWeight --binStart=2 --binEnd=100 --model_def='resnethybridv318' --name='sketchy_resnethybridv318' | tee "textlogs/sketchy_resnethybridv318.txt"
