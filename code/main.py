import os
import torch.backends.cudnn as cudnn
import opts
import train
import utils
import models.__init__ as init
import datasets.__datainit__ as init_data
#from tensorboard_logger import Logger

parser = opts.myargparser()

def main():
    global opt, best_prec1

    opt = parser.parse_args()
    opt.logdir = opt.logdir+'/'+opt.name
    logger = 'hi'

    best_prec1 = 0
    print(opt)

    # Initialize the model, criterion and the optimizer
    model = init.load_model(opt)
    model, criterion, optimizer = init.setup(model,opt)
    # Display the model structure
    print(model)

    # Setup trainer and validation
    trainer = train.Trainer(model, criterion, optimizer, opt, logger)
    validator = train.Validator(model, criterion, opt, logger)

    # Load model from a checkpoint if mentioned in opts
    if opt.resume:
        if os.path.isfile(opt.resume):
            model, optimizer, opt, best_prec1 = init.resumer(opt, model, optimizer)
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    cudnn.benchmark = True

    # Setup the train and validation data loaders
    dataloader = init_data.load_data(opt)
    train_loader = dataloader.train_loader
    val_loader = dataloader.val_loader

    for epoch in range(opt.start_epoch, opt.epochs):
        utils.adjust_learning_rate(opt, optimizer, epoch)
        print("Starting epoch number:",epoch+1,"Learning rate:", optimizer.param_groups[0]["lr"])

        if opt.testOnly == False:
            # Train the network over the training data
            trainer.train(train_loader, epoch, opt)

        #if opt.tensorboard:
            #logger.scalar_summary('learning_rate', opt.lr, epoch)

        # Measure the validation accuracy
        acc = validator.validate(val_loader, epoch, opt)
        best_prec1 = max(acc, best_prec1)
        if best_prec1 == acc:
            # Save the new model if the accuracy is better than the previous saved model
            init.save_checkpoint(opt, model, optimizer, best_prec1, epoch)

        print('Best accuracy: [{0:.3f}]\t'.format(best_prec1))

if __name__ == '__main__':
    main()
