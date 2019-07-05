# Code for - Hybrid Binary Networks: Optimizing for Accuracy, Efficiency and Memory (WACV 18)

Binarization is an extreme network compression approach that provides large computational speedups along with energy and memory savings, albeit at significant accuracy costs. We investigate the question of where to binarize inputs at layer-level granularity and show that selectively binarizing the inputs to specific layers in the network could lead to significant improvements in accuracy while preserving most of the advantages of binarization. We analyze the binarization tradeoff using a metric that jointly models the input binarization-error and computational cost and introduce an efficient algorithm to select layers whose inputs are to be binarized. Practical guidelines based on insights obtained from applying the algorithm to a variety of models are discussed.

### Cite
If you use our paper or repo in your work, please cite the original paper as:
```
@article{Huang2016Densely,
  author  = {Ameya Prabhu, Vishal Batchu, Rohit Gajawada, Sri Aurobindo Munagala, Anoop Namboodiri},
  title   = {Hybrid Binary Networks: Optimizing for Accuracy, Efficiency and Memory},
  journal = {IEEE Winter Conference on Applications of Computer Vision (WACV)},
  year    = {2018}
}
```

### Usage instructions
* Clone the repo
* Install PyTorch and other required dependencies
* Run using,
`python3 main.py --dataset='<dataset>' --data_dir='<data directory path>' --nClasses=<num_classes> --workers=8 --epochs=<num_epochs> --batch-size=<batch_size> --testbatchsize=<test_batch_size> --learningratescheduler='<lr_scheduler>' --decayinterval=12 --decaylevel=2 --optimType='<optimizer>' --verbose --maxlr=<maximum_lr> --minlr=<minimum_lr> --binStart=<bin_start> --binEnd=<bin_end> --binaryWeight --weightDecay=0 --model_def='<model_name>' --inpsize=<input_size> --name='<experiment_name>'`
* You could also take a look at the existing scripts for samples
