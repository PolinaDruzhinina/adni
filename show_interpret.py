"""The project is inspired by the clinica /clinicadl library, the code is taken from https://github.com/aramis-lab/AD-DL"""

from os import path
import sys
import torch
import argparse
import pandas as pd
import numpy as np
import nibabel as nib

from torch.utils.data import DataLoader

from data import load_data, get_transforms, MRIDatasetImage, generate_sampler
from model import Conv5_FC3
from utils import return_logger, display_interpretation
from Interpretation.grad_cam import get_masks

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', help='path to output dir', type=str, default='/home/ADNI/image_cnn')
parser.add_argument('--input_dir', help='path to input dir', type=str, default='/data/caps')
parser.add_argument('--tsv_path', help='path', type=str, default='/home/ADNI/data_info/labels_lists_new')
parser.add_argument('--task', help='train/val or test', type=str, default='test')
parser.add_argument("--diagnoses", help="Labels that must be extracted from merged_tsv.",
                    nargs="+", type=str, choices=['AD', 'BV', 'CN', 'MCI', 'sMCI', 'pMCI'], default=['AD', 'CN'])
parser.add_argument("--interp_path", help="Path to interpretation img", type=str, default=None)
parser.add_argument("--item", help="Item in dataset to interp", type=int, default=0)
parser.add_argument("--cut_coord", help="Cut coord", type=str, default=(40, 25, 55))
parser.add_argument("--name", help="Name interp img", type=str, default='mean')
parser.add_argument("--threshold", help="threshold", type=float, default=0.35)
parser.add_argument("--baseline", action="store_true", default=False,
                    help="If provided, only the baseline sessions are used for training.")
parser.add_argument('--fold', default=0, type=int, help='Num of split')
parser.add_argument('--n_splits', default=5, type=int, help='n splits for training')
parser.add_argument('--preprocessing', help='Defines the type of preprocessing of CAPS data.',
                    choices=['t1-linear', 't1-extensive', 't1-volume'], type=str, default='t1-linear')
parser.add_argument('-b', '--batch_size', default=4, type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=0, type=int, help='Number of workers used in dataloading')
parser.add_argument('--mode', type=str, choices=['image', 'slice'], default='image')
parser.add_argument('--minmaxnormalization', default=True, help='MinMaxNormalization')
parser.add_argument('--data_augmentation', default=None, help='Augmentation')
parser.add_argument('--verbose', '-v', action='count', default=0)
args = parser.parse_args()

sys.stdout.flush()
if __name__ == '__main__':
    args = parser.parse_args()

    logger = return_logger(args.verbose, "Logger")
    train_transforms, all_transforms = get_transforms(args.mode,
                                                      minmaxnormalization=args.minmaxnormalization,
                                                      data_augmentation=args.data_augmentation)

    training_df, valid_df = load_data(args.tsv_path, args.diagnoses, args.fold, n_splits=args.n_splits,
                                          baseline=args.baseline, logger=logger)

    if args.task == 'test':
        test_df = pd.DataFrame()
        logger.debug("Test path %s" % args.tsv_path)

        for diagnosis in args.diagnoses:
            test_diagnosis_path = path.join(
            args.tsv_path, diagnosis + '_baseline.tsv')

            test_diagnosis_df = pd.read_csv(test_diagnosis_path, sep='\t')
            test_df = pd.concat([test_df, test_diagnosis_df])

        test_df.reset_index(inplace=True, drop=True)
        test_df["cohort"] = "single"

        data_test = MRIDatasetImage(args.input_dir, data_df=test_df, preprocessing=args.preprocessing,
                                        train_transformations=train_transforms, all_transformations=all_transforms,
                                        labels=True)
        test_loader = DataLoader(data_test, batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.num_workers, pin_memory=True)
        dataset_img = data_test.__getitem__(args.item)
        print(data_test.df)
        print(dataset_img['label'])
        print(dataset_img['image_path'])
        #interp_img = np.load(args.interp_path)
        interp_img = np.array(nib.load(args.interp_path).dataobj)
        print(interp_img.shape)
        display_interpretation(interp_img, dataset_img['image'],name = args.name)
    elif args.task == 'train':
        data_train = MRIDatasetImage(args.input_dir, data_df=training_df, preprocessing=args.preprocessing,
                                         train_transformations=train_transforms, all_transformations=all_transforms,
                                         labels=True)

        train_sampler = generate_sampler(data_train)

        train_loader = DataLoader(data_train, batch_size=args.batch_size, sampler=train_sampler,
                                      num_workers=args.num_workers, pin_memory=True)
        dataset_img = train_loader.datasets.__getitem__(args.item)
        print(dataset_img['label'])
        interp_img = np.load(args.interp_path)
        print(interp_img.shape)
        display_interpretation(interp_img, dataset_img)

    elif args.task == 'val':
        data_valid = MRIDatasetImage(args.input_dir, data_df=valid_df, preprocessing=args.preprocessing,
                                         train_transformations=train_transforms, all_transformations=all_transforms,
                                         labels=True)
        valid_loader = DataLoader(data_valid, batch_size=args.batch_size, shuffle=False,
                                      num_workers=args.num_workers, pin_memory=True)
        dataset_img = valid_loader.datasets.__getitem__(args.item)
        print(dataset_img['label'])
        interp_img = np.load(args.interp_path)
        print(interp_img.shape)
        display_interpretation(interp_img, dataset_img)


