import os
import sys
import torch
import argparse
import shutil
import logging
from time import time
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader

from model import AutoEncoder, Conv5_FC3, transfer_autoencoder_weights, transfer_cnn_weights
from data import load_data, generate_sampler, get_transforms, MRIDatasetImage
from utils import return_logger, EarlyStopping
from test import test, save_checkpoint, test_single_cnn
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--project_name', help='project name for W&B', type=str, default='ADNI')
parser.add_argument('--output_dir', help='path to output dir', type=str, default='/home/ADNI/image_cnn_mci_cn')
parser.add_argument('--input_dir', help='path to input dir', type=str, default='/data/caps')
parser.add_argument('--tsv_path', help='path', type=str, default='/home/ADNI/data_info/labels_lists_new/train')
parser.add_argument('--transfer_learning_path', help='transfer_learning_path', type=str, default=None)
#parser.add_argument('--transfer_learning_path', help='transfer_learning_path', type=str, default='/data/results'
#                                                                                                 '/image_autoencoder')
parser.add_argument("--diagnoses", help="Labels that must be extracted from merged_tsv.",
                    nargs="+", type=str, choices=['AD', 'BV', 'CN', 'MCI', 'sMCI', 'pMCI'], default=['MCI', 'CN'])
parser.add_argument("--baseline", action="store_true", default=False,
                    help="If provided, only the baseline sessions are used for training.")
parser.add_argument('--n_splits', default=5, type=int, help='n splits for training')
parser.add_argument('--preprocessing', help='Defines the type of preprocessing of CAPS data.',
                    choices=['t1-linear', 't1-extensive', 't1-volume'], type=str, default='t1-linear')
# parser.add_argument('--prepare_dl', help='''If True the extract slices or patche are used, otherwise the they
#                 will be extracted on the fly (if necessary).''', default=False, action="store_true")
parser.add_argument('-b', '--batch_size', default=4, type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=0, type=int, help='Number of workers used in dataloading')
parser.add_argument('--gpu', default=True, type=bool, help='Use cuda to train model')
parser.add_argument('--id_gpu', default=3, type=int, help="Id of gpu device")
parser.add_argument('--learning_rate', default=1e-4, type=float, help='initial learning rate')
parser.add_argument('--dropout', default=0.5, type=float, help='initial dropout')
parser.add_argument('--epochs', default=20, type=int, help='max epoch for training')
parser.add_argument('--save_folder', default='img/', help='Location to save checkpoint models')
parser.add_argument('--mode', type=str, choices=['image', 'slice'], default='image')
parser.add_argument('--model', help='model', type=str, default='Conv5_FC3')
parser.add_argument('--tl_selection', help='transfer learning selection', type=str, default='best_loss')
parser.add_argument('--mode_task', help='transfer learning from model', type=str, default='autoencoder')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay')
parser.add_argument('--momentum', default=0.999, type=float, help='momentum')
parser.add_argument('--tolerance', default=0, type=float, help='tolerance')
parser.add_argument('--patience', default=5, type=float, help='patience')
parser.add_argument('--accumulation_steps', default=1, type=int, help='Accumulates gradients during the given '
                                                                      'number of iterations before performing the '
                                                                      'weight update in order to virtually increase '
                                                                      'the size of the batch.')
parser.add_argument('--evaluation_steps', default=0, type=int,
                    help='Fix the number of iterations to perform before computing an evaluations. Default will only '
                         'perform one evaluation at the end of each epoch.')
parser.add_argument('--minmaxnormalization', default=True, help='MinMaxNormalization')
parser.add_argument('--data_augmentation', default=None, help='Augmentation')
parser.add_argument('--verbose', '-v', action='count', default=0)
parser.add_argument('--autoencoder', default=False, type=bool)

args = parser.parse_args()

sys.stdout.flush()

os.environ["WANDB_API_KEY"] = 'e42d0e4a9e1aeb4b9dab719ba8c7e39a0e0a1c7e'


def check_and_clean(d):
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)


def train(model, train_loader, valid_loader, criterion, optimizer, resume, log_dir, model_dir, options, logger=None):
    """
    Function used to train a CNN.
    The best model and checkpoint will be found in the 'best_model_dir' of options.output_dir.
    Args:
        model: (Module) CNN to be trained
        train_loader: (DataLoader) wrapper of the training dataset
        valid_loader: (DataLoader) wrapper of the validation dataset
        criterion: (loss) function to calculate the loss
        optimizer: (torch.optim) optimizer linked to model parameters
        resume: (bool) if True, a begun job is resumed
        log_dir: (str) path to the folder containing the logs
        model_dir: (str) path to the folder containing the models weights and biases
        options: (Namespace) ensemble of other options given to the main script.
        logger: (logging object) writer to stdout and stderr
    """
    from time import time

    if logger is None:
        logger = logging
    wandb.watch(model, criterion, log = "all", log_freq=5)

    columns = ['epoch', 'iteration', 'time',
               'balanced_accuracy_train', 'loss_train',
               'balanced_accuracy_valid', 'loss_valid']
    if hasattr(model, "variational") and model.variational:
        columns += ["kl_loss_train", "kl_loss_valid"]
    filename = os.path.join(os.path.dirname(log_dir), 'training.tsv')

    if not resume:
        check_and_clean(model_dir)
        check_and_clean(log_dir)

        results_df = pd.DataFrame(columns=columns)
        with open(filename, 'w') as f:
            results_df.to_csv(f, index=False, sep='\t')
        options.beginning_epoch = 0

    else:
        if not os.path.exists(filename):
            raise ValueError('The training.tsv file of the resumed experiment does not exist.')
        truncated_tsv = pd.read_csv(filename, sep='\t')
        truncated_tsv.set_index(['epoch', 'iteration'], inplace=True)
        truncated_tsv.drop(options.beginning_epoch, level=0, inplace=True)
        truncated_tsv.to_csv(filename, index=True, sep='\t')

    # Initialize variables
    best_valid_accuracy = -1.0
    best_valid_loss = np.inf
    epoch = options.beginning_epoch

    model.train()  # set the model to training mode
    train_loader.dataset.train()

    early_stopping = EarlyStopping('min', min_delta=options.tolerance, patience=options.patience)
    mean_loss_valid = None
    t_beginning = time()

    while epoch < options.epochs and not early_stopping.step(mean_loss_valid):
        logger.info("Beginning epoch %i." % epoch)
        
        print(epoch) 
        model.zero_grad()
        evaluation_flag = True
        step_flag = True
        tend = time()
        total_time = 0

        for i, data in enumerate(train_loader, 0):
            
            t0 = time()
            total_time = total_time + t0 - tend
            if options.gpu:
                imgs, labels = data['image'].cuda(), data['label'].cuda()
            else:
                imgs, labels = data['image'], data['label']
            
            train_output = model(imgs)
            
            loss = criterion(train_output, labels)

            # Back propagation
            loss.backward()

            del imgs, labels

            if (i + 1) % options.accumulation_steps == 0:
              
                step_flag = False
                optimizer.step()
                optimizer.zero_grad()

                del loss

                # Evaluate the model only when no gradients are accumulated
                if options.evaluation_steps != 0 and (i + 1) % options.evaluation_steps == 0:
                    evaluation_flag = False
                    
                    _, results_train = test(model, train_loader, options.gpu, criterion)
                    mean_loss_train = results_train["total_loss"] / (len(train_loader) * train_loader.batch_size)

                    _, results_valid = test(model, valid_loader, options.gpu, criterion)
                    mean_loss_valid = results_valid["total_loss"] / (len(valid_loader) * valid_loader.batch_size)
                    model.train()
                    train_loader.dataset.train()

                    global_step = i + epoch * len(train_loader)
                    wandb.log({"train": {"epoch": epoch, 'balanced_accuracy': results_train["balanced_accuracy"], 'loss': mean_loss_train},
                    "test": {'balanced_accuracy': results_valid[
                                                      "balanced_accuracy"], 'loss': mean_loss_valid}}, step = global_step)

                    logger.info("%s level training accuracy is %f at the end of iteration %d"
                                % (options.mode, results_train["balanced_accuracy"], i))
                    logger.info("%s level validation accuracy is %f at the end of iteration %d"
                                % (options.mode, results_valid["balanced_accuracy"], i))

                    t_current = time() - t_beginning
                    row = [epoch, i, t_current,
                           results_train["balanced_accuracy"], mean_loss_train,
                           results_valid["balanced_accuracy"], mean_loss_valid]
                    if hasattr(model, "variational") and model.variational:
                        row += [results_train["total_kl_loss"] / (len(train_loader) * train_loader.batch_size),
                                results_valid["total_kl_loss"] / (len(valid_loader) * valid_loader.batch_size)]
                    row_df = pd.DataFrame([row], columns=columns)
                    with open(filename, 'a') as f:
                        row_df.to_csv(f, header=False, index=False, sep='\t')

            tend = time()
           
        logger.debug('Mean time per batch loading: %.10f s'
                     % (total_time / len(train_loader) * train_loader.batch_size))

        # If no step has been performed, raise Exception
        if step_flag:
            raise Exception('The model has not been updated once in the epoch. The accumulation step may be too large.')


        # Always test the results and save them once at the end of the epoch
        model.zero_grad()
        logger.debug('Last checkpoint at the end of the epoch %d' % epoch)
        
        _, results_train = test(model, train_loader, options.gpu, criterion)
        mean_loss_train = results_train["total_loss"] / (len(train_loader) * train_loader.batch_size)

        _, results_valid = test(model, valid_loader, options.gpu, criterion)
        mean_loss_valid = results_valid["total_loss"] / (len(valid_loader) * valid_loader.batch_size)
        model.train()
        train_loader.dataset.train()

        global_step = (epoch + 1) * len(train_loader)
        wandb.log({"train": {'epoch': epoch, 'balanced_accuracy': results_train["balanced_accuracy"], 'loss': mean_loss_train},
            "test": {'epoch': epoch, 'balanced_accuracy': results_valid["balanced_accuracy"],'loss' : mean_loss_valid}}, step = global_step)

        logger.info("%s level training accuracy is %f at the end of iteration %d"
                    % (options.mode, results_train["balanced_accuracy"], len(train_loader)))
        logger.info("%s level validation accuracy is %f at the end of iteration %d"
                    % (options.mode, results_valid["balanced_accuracy"], len(train_loader)))

        t_current = time() - t_beginning
        row = [epoch, i, t_current,
               results_train["balanced_accuracy"], mean_loss_train,
               results_valid["balanced_accuracy"], mean_loss_valid]
        if hasattr(model, "variational") and model.variational:
            row += [results_train["total_kl_loss"] / (len(train_loader) * train_loader.batch_size),
                    results_valid["total_kl_loss"] / (len(valid_loader) * valid_loader.batch_size)]
        row_df = pd.DataFrame([row], columns=columns)
        with open(filename, 'a') as f:
            row_df.to_csv(f, header=False, index=False, sep='\t')

        accuracy_is_best = results_valid["balanced_accuracy"] > best_valid_accuracy
        loss_is_best = mean_loss_valid < best_valid_loss
        best_valid_accuracy = max(results_valid["balanced_accuracy"], best_valid_accuracy)
        best_valid_loss = min(mean_loss_valid, best_valid_loss)

        save_checkpoint({'model': model.state_dict(),
                         'epoch': epoch,
                         'valid_loss': mean_loss_valid,
                         'valid_acc': results_valid["balanced_accuracy"]},
                        accuracy_is_best, loss_is_best,
                        model_dir)
        # Save optimizer state_dict to be able to reload
        save_checkpoint({'optimizer': optimizer.state_dict(),
                         'epoch': epoch,
                         'name': 'Adam',
                         },
                        False, False,
                        model_dir,
                        filename='optimizer.pth.tar')

        epoch += 1
        
    os.remove(os.path.join(model_dir, "optimizer.pth.tar"))
    os.remove(os.path.join(model_dir, "checkpoint.pth.tar"))


def train_single_cnn(args):

    main_logger = return_logger(args.verbose, "main process")
    train_logger = return_logger(args.verbose, "train")
    eval_logger = return_logger(args.verbose, "final evaluation")
    check_and_clean(args.output_dir)
    train_transforms, all_transforms = get_transforms(args.mode,
                                                      minmaxnormalization=args.minmaxnormalization,
                                                      data_augmentation=args.data_augmentation)
    fold_iterator = range(args.n_splits)
    for fi in fold_iterator:
        print("Fold %i" % fi)
        wandb.init(project=args.project_name, group="adni_mci_cn_kf5", job_type='Kfold_' + str(fi), reinit=True)
        wandb.config.update(args)
        params = wandb.config
        training_df, valid_df = load_data(params.tsv_path, params.diagnoses, fi, n_splits=params.n_splits,
                                          baseline=params.baseline, logger=main_logger)

        data_train = MRIDatasetImage(params.input_dir, data_df=training_df, preprocessing=params.preprocessing,
                                     train_transformations=train_transforms, all_transformations=all_transforms,
                                     labels=True)
        data_valid = MRIDatasetImage(params.input_dir, data_df=valid_df, preprocessing=params.preprocessing,
                                     train_transformations=train_transforms, all_transformations=all_transforms,
                                     labels=True)

        train_sampler = generate_sampler(data_train)

        train_loader = DataLoader(data_train, batch_size=params.batch_size, sampler=train_sampler,
                                  num_workers=params.num_workers, pin_memory=True)

        valid_loader = DataLoader(data_valid, batch_size=params.batch_size, shuffle=False,
                                  num_workers=params.num_workers, pin_memory=True)

        # Initialize the model
        main_logger.info('Initialization of the model')
        if params.id_gpu is not None:
            torch.cuda.set_device(params.id_gpu)

        model = eval(params.model)(dropout=params.dropout)
        if params.gpu:
            model.cuda()
        else:
            model.cpu()
        if params.autoencoder:
            
            model = AutoEncoder(model)
            
        if params.transfer_learning_path is not None:
        
        
            if params.mode_task == "autoencoder":
            
                main_logger.info("A pretrained autoencoder is loaded at path %s" % params.transfer_learning_path)
                model = transfer_autoencoder_weights(model, params.transfer_learning_path, fi)
            

            else:
                main_logger.info("A pretrained CNN is loaded at path %s" % params.transfer_learning_path)
                model = transfer_cnn_weights(model, params.transfer_learning_path, fi, selection=params.tl_selection,
                                         cnn_index=None)
        else:
            main_logger.info("The model is trained from scratch")

            
        if params.gpu:
            model.cuda()
        else:
            model.cpu()
        # Define criterion and optimizer
        criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                                     lr=params.learning_rate,
                                     weight_decay=params.weight_decay)

        # Define output directories
        log_dir = os.path.join(
            params.output_dir, 'fold-%i' % fi, 'logs')
        model_dir = os.path.join(
                params.output_dir, 'fold-%i' % fi, 'models')

        main_logger.debug('Beginning the training task')
        train(model, train_loader, valid_loader, criterion,
              optimizer, False, log_dir, model_dir, params, train_logger)

        test_single_cnn(model, params.output_dir, train_loader, "train",
                        fi, criterion, params.mode, eval_logger, selection_threshold=None, gpu=params.gpu)
        test_single_cnn(model, params.output_dir, valid_loader, "validation",
                        fi, criterion, params.mode, eval_logger, selection_threshold=None, gpu=params.gpu)
        wandb.finish()


if __name__ == '__main__':
    args = parser.parse_args()

    train_single_cnn(args)

