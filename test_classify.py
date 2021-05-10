import os
from os import path
import sys
import torch
import argparse
import shutil
import logging
from time import time
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader

from model import Conv5_FC3
from data import load_data,  get_transforms, MRIDatasetImage
from utils import return_logger
from test import  test_single_cnn


parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', help='path to output dir', type=str, default='/home/ADNI/image_cnn')
parser.add_argument('--input_dir', help='path to input dir', type=str, default='/data/caps')
parser.add_argument('--tsv_path_test', help='path', type=str, default='/home/ADNI/data_info/labels_lists_new/test')
parser.add_argument("--diagnoses", help="Labels that must be extracted from merged_tsv.",
                    nargs="+", type=str, choices=['AD', 'BV', 'CN', 'MCI', 'sMCI', 'pMCI'], default=['AD', 'CN'])
parser.add_argument("--baseline", action="store_true", default=False,
                    help="If provided, only the baseline sessions are used for training.")
parser.add_argument('--n_splits', default=5, type=int, help='n splits for training')
parser.add_argument('--preprocessing', help='Defines the type of preprocessing of CAPS data.',
                    choices=['t1-linear', 't1-extensive', 't1-volume'], type=str, default='t1-linear')
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
parser.add_argument('--minmaxnormalization', default=True, help='MinMaxNormalization')
parser.add_argument('--data_augmentation', default=None, help='Augmentation')
parser.add_argument('--verbose', '-v', action='count', default=0)
args = parser.parse_args()

sys.stdout.flush()


if __name__ == '__main__':
    args = parser.parse_args()

    test_logger = return_logger(args.verbose, "test logger")
    train_transforms, all_transforms = get_transforms(args.mode,
                                                      minmaxnormalization=args.minmaxnormalization,
                                                      data_augmentation=args.data_augmentation)

    fold_iterator = range(args.n_splits)
    for fi in fold_iterator:
        if test_logger is None:
            logger = test_logger

        test_df = pd.DataFrame()

        test_path = path.join(args.tsv_path_test, 'test')

        logger.debug("Train path %s" % test_path)

        for diagnosis in args.diagnoses:
            test_diagnosis_path = path.join(
                test_path, diagnosis + '_baseline.tsv')

            test_diagnosis_df = pd.read_csv(test_diagnosis_path, sep='\t')

            test_df = pd.concat([test_df, test_diagnosis_df])

        test_df.reset_index(inplace=True, drop=True)
        test_df["cohort"] = "single"

        data_test = MRIDatasetImage(args.input_dir, data_df=test_df, preprocessing=args.preprocessing,
                                 train_transformations=train_transforms, all_transformations=all_transforms,
                                 labels=True)

        test_loader = DataLoader(data_test, batch_size=args.batch_size, shuffle = False,
                              num_workers=args.num_workers, pin_memory=True)


        if args.id_gpu is not None:
            torch.cuda.set_device(args.id_gpu)

        model = eval(args.model)(dropout=args.dropout)
        if args.gpu:
            model.cuda()
        else:
            model.cpu()

        criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                                     lr=args.learning_rate,
                                     weight_decay=args.weight_decay)

        test_single_cnn(model, args.output_dir, test_loader, "test",
                    fi, criterion, args.mode, test_logger, selection_threshold=None, gpu=args.gpu)