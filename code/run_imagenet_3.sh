#!/bin/bash
#SBATCH -A research
#SBATCH -n 24
#SBATCH --gres=gpu:3
#SBATCH --mem-per-cpu=4000
#SBATCH --time=72:00:00
#SBATCH --nodelist=gnode24
module add cuda/8.0
module add cudnn/7-cuda-8.0

array[0]=`echo $CUDA_VISIBLE_DEVICES | cut -d"," -f1`
array[1]=`echo $CUDA_VISIBLE_DEVICES | cut -d"," -f2`
array[2]=`echo $CUDA_VISIBLE_DEVICES | cut -d"," -f3`

bash clean.sh; CUDA_VISIBLE_DEVICES="${array[0]}" python3 main.py --dataset='imagenet12' --data_dir='/ssd_scratch/cvit/Imagenet12' --nClasses=1000 --workers=8  --epochs=90 --batch-size=512 --testbatchsize=32 --learningratescheduler='imagenetscheduler' --decayinterval=12 --decaylevel=2 --optimType='adam' --verbose --maxlr=8e-5 --minlr=1e-5 --binStart=2 --binEnd=7 --binaryWeight --weightDecay=0 --model_def='alexnethybrid' --inpsize=227 --name='imagenet_alexnethybrid_cameraready2' | tee "textlogs/imagenet_alexnethybrid_cameraready2.txt" &
bash clean.sh; CUDA_VISIBLE_DEVICES="${array[1]}" python3 main.py --dataset='imagenet12' --data_dir='/ssd_scratch/cvit/Imagenet12' --nClasses=1000 --workers=8  --epochs=90 --batch-size=512 --testbatchsize=32 --learningratescheduler='imagenetscheduler' --decayinterval=12 --decaylevel=2 --optimType='sgd' --verbose --maxlr=8e-5 --minlr=1e-5 --binStart=2 --binEnd=7 --binaryWeight --weightDecay=0 --model_def='alexnetwbin' --inpsize=227 --name='imagenet_alexnetwbin_cameraready2' | tee "textlogs/imagenet_alexnetwbin_cameraready2.txt" &
bash clean.sh; CUDA_VISIBLE_DEVICES="${array[2]}" python3 main.py --dataset='imagenet12' --data_dir='/ssd_scratch/cvit/Imagenet12' --nClasses=1000 --workers=8  --epochs=90 --batch-size=512 --testbatchsize=32 --learningratescheduler='imagenetscheduler' --decayinterval=20 --decaylevel=2 --optimType='adam' --verbose --maxlr=8e-5 --minlr=1e-5 --binStart=2 --binEnd=7 --binaryWeight --weightDecay=0 --model_def='alexnetfbin' --inpsize=227 --name='imagenet_alexnetfbin_cameraready2' | tee "textlogs/imagenet_alexnetfbin_cameraready2.txt"
