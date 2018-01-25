import argparse

# Dataset to use
dset_choices = ['tuberlin','cifar10','imagenet12','cifar100','sketchyrecognition']
# Reporting
reporttype_choices = ['acc','error']
# Criterion to train the network (Not all implemented)
criterion_choices = ['nllLoss','tripletmargin', 'mse', 'hingeEmbedding', 'kldiv']
# Optimizer choices (Not all implemented)
optim_choices = ['sgd','adam','adagrad', 'adamax', 'adadelta']
# Learning rate schedules
lrscheduler_choices = ['decayscheduler', 'imagenetscheduler']
# Model to train
model_def_choices = ['alexnet', 'alexnetfbin', 'alexnethybrid', 'alexnethybridv2', 'alexnetwbin',
                    'googlenet', 'googlenetfbin', 'googlenet_expanded', 'googlenetwbin',
                    'mobilenet',
                    'resnet18', 'resnetfbin18', 'resnethybrid18', 'resnethybridv218', 'resnethybridv318', 'resnetwbin18',
                    'sketchanet', 'sketchanetfbin', 'sketchanethybrid', 'sketchanethybridv2', 'sketchanetwbin',
                    'squeezenet', 'squeezenetfbin', 'squeezenethybrid', 'squeezenethybridv2', 'squeezenethybridv3', 'squeezenetwbin',
                    'vgg16_bncifar', 'densenet', 'bincifar']

def myargparser():
    parser = argparse.ArgumentParser(description='PyTorch Core Training')

    parser.add_argument('--dataset', required=True, type=str, choices=dset_choices, help='chosen dataset')
    parser.add_argument('--nClasses', required=True, type=int, help='chosen dataset')
    parser.add_argument('--data_dir', required=True, type=str, help='chosen data directory')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')

    parser.add_argument('--epochs', required=True, type=int, help='number of total epochs to run')
    parser.add_argument('--batch-size', required=True, type=int, help='mini-batch size (default: 128)')
    parser.add_argument('--testbatchsize', required=True, type=int, help='input batch size for testing (default: 1000)')
    parser.add_argument('--tenCrop', action='store_true', help='ten-crop testing')
    parser.add_argument('--printfreq', default=200, type=int, help='print frequency (default: 10)')
    parser.add_argument('--learningratescheduler', required=True, type=str, help='LR Scheduler. Options:'+str(lrscheduler_choices))

    parser.add_argument('--decayinterval', type=int, help='decays by a power of decay_var in these epochs')
    parser.add_argument('--decaylevel', type=int, help='decays by a power of decaylevel')
    parser.add_argument('-criterion', default="nllLoss", choices=criterion_choices, type=str, help='Criterion. Options:'+str(criterion_choices))
    parser.add_argument('--optimType', required=True, choices=optim_choices, type=str, help='Optimizers. Options:'+str(optim_choices))

    parser.add_argument('--maxlr', required=True, type=float, help='initial learning rate')
    parser.add_argument('--lr', type=float, help='initial learning rate')
    parser.add_argument('--minlr', required=True, type=float, help='initial learning rate')

    parser.add_argument('--nesterov', action='store_true', help='nesterov momentum')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum (Default: 0.9)')
    parser.add_argument('--weightDecay', default=0, type=float, help='weight decay (Default: 1e-4)')

    parser.add_argument('--model_def', required=True, choices=model_def_choices, help='Architectures to be loaded. Options:'+str(model_def_choices))
    parser.add_argument('--inpsize', default=224, type=int, help='Architectures to be loaded. Options:'+str(model_def_choices))
    parser.add_argument('--weight_init', action='store_false', help='Turns off weight inits')
    parser.add_argument('--name', required=True, type=str,help='name of experiment')


    parser.add_argument('--cachemode', default=True, help='if cachemode')
    parser.add_argument('--cuda',  default=True, help='if cuda is available')
    parser.add_argument('--manualSeed',  default=123, help='fixed seed for experiments')
    parser.add_argument('--ngpus',  default=1, help='no. of gpus')
    parser.add_argument('--logdir',  type=str, default='../logs/', help='log directory')
    parser.add_argument('--tensorboard',help='Log progress to TensorBoard', default=True)
    parser.add_argument('--testOnly', default=False, type=bool, help='run on validation set only')
    parser.add_argument('--start-epoch', default=0, type=int,help='manual epoch number (useful on restarts)')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
    parser.add_argument('--pretrained_file', default="")

    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--store', default='', type=str,
                        help='path to storing checkpoints (default: none)')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')

    parser.add_argument('--binStart', type=int)
    parser.add_argument('--binEnd', type=int)
    parser.add_argument('--binaryWeight', action='store_true')
    parser.add_argument('--bnnModel', action='store_true')
    parser.add_argument('--calculateBinarizationLosses', action='store_true', help='Do you want to compute activation losses')

    return parser
