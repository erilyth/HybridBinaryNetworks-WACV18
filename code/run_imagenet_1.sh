#!/bin/bash
#SBATCH -A research
#SBATCH -n 32
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=4000
#SBATCH --time=72:00:00
#SBATCH --nodelist=gnode23
module add cuda/8.0
module add cudnn/7-cuda-8.0

array[0]=`echo $CUDA_VISIBLE_DEVICES | cut -d"," -f1`
array[1]=`echo $CUDA_VISIBLE_DEVICES | cut -d"," -f2`
array[2]=`echo $CUDA_VISIBLE_DEVICES | cut -d"," -f3`
array[3]=`echo $CUDA_VISIBLE_DEVICES | cut -d"," -f4`

bash clean.sh; CUDA_VISIBLE_DEVICES="${array[0]}" python3 main.py --dataset='imagenet12' --data_dir='/ssd_scratch/cvit/Imagenet12' --nClasses=1000 --workers=8  --epochs=90 --batch-size=512 --testbatchsize=32 --learningratescheduler='imagenetscheduler' --decayinterval=12 --decaylevel=2 --optimType='sgd' --verbose --maxlr=0.0001 --minlr=0.00001 --binStart=2 --binEnd=7 --binaryWeight --weightDecay=0 --model_def='alexnethybrid' --inpsize=224 --name='imagenet_alexnethybrid_cameraready1' | tee "textlogs/imagenet_alexnethybrid_cameraready1.txt" &
bash clean.sh; CUDA_VISIBLE_DEVICES="${array[1]}" python3 main.py --dataset='imagenet12' --data_dir='/ssd_scratch/cvit/Imagenet12' --nClasses=1000 --workers=8  --epochs=90 --batch-size=512 --testbatchsize=32 --learningratescheduler='imagenetscheduler' --decayinterval=12 --decaylevel=2 --optimType='sgd' --verbose --maxlr=0.0001 --minlr=0.00001 --binStart=2 --binEnd=7 --binaryWeight --weightDecay=0 --model_def='alexnethybridv2' --inpsize=224 --name='imagenet_alexnethybridv2_cameraready1' | tee "textlogs/imagenet_alexnethybridv2_cameraready1.txt" &
bash clean.sh; CUDA_VISIBLE_DEVICES="${array[2]}" python3 main.py --dataset='imagenet12' --data_dir='/ssd_scratch/cvit/Imagenet12' --nClasses=1000 --workers=8  --epochs=90 --batch-size=512 --testbatchsize=32 --learningratescheduler='imagenetscheduler' --decayinterval=12 --decaylevel=2 --optimType='sgd' --verbose --maxlr=0.0001 --minlr=0.00001 --binStart=2 --binEnd=7 --binaryWeight --weightDecay=0 --model_def='alexnetwbin' --inpsize=224 --name='imagenet_alexnetwbin_cameraready1' | tee "textlogs/imagenet_alexnetwbin_cameraready1.txt" &
bash clean.sh; CUDA_VISIBLE_DEVICES="${array[3]}" python3 main.py --dataset='imagenet12' --data_dir='/ssd_scratch/cvit/Imagenet12' --nClasses=1000 --workers=8  --epochs=90 --batch-size=512 --testbatchsize=32 --learningratescheduler='imagenetscheduler' --decayinterval=20 --decaylevel=2 --optimType='sgd' --verbose --maxlr=0.0001 --minlr=0.00001 --binStart=2 --binEnd=7 --binaryWeight --weightDecay=0 --model_def='alexnetfbin' --inpsize=224 --name='imagenet_alexnetfbin_cameraready1' | tee "textlogs/imagenet_alexnetfbin_cameraready1.txt"
