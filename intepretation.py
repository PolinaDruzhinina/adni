"""The project is inspired by the clinica /clinicadl library, the code is taken from https://github.com/aramis-lab/AD-DL"""

from os import path
import sys
import torch
import argparse
import pandas as pd

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
parser.add_argument("--mask_type", help="Type of interpretation",
                    type=str, choices=['grad_cam', 'guided_backprop', 'mean_pertrub'], default='grad_cam')
parser.add_argument('--fold', default=0, type=int, help='Num of split')
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
parser.add_argument('--mode', type=str, choices=['image', 'slice'], default='image')
parser.add_argument('--model', help='model', type=str, default='Conv5_FC3')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay')
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

    if logger is None:
        logger = logger

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
    training_df, valid_df = load_data(args.tsv_path, args.diagnoses,args.fold, n_splits=args.n_splits,
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
        get_masks(model, test_loader, args.fold, args.output_dir, mean_mask=True, mask_type=args.mask_type,
                  size=data_test.size, task=args.task)
        # np.save(os.path.join(CHECKPOINTS_DIR, 'masks_grad_cam_part1_for_labels_0'), masks_grad)
    elif args.task == 'train':

        data_train = MRIDatasetImage(args.input_dir, data_df=training_df, preprocessing=args.preprocessing,
                                     train_transformations=train_transforms, all_transformations=all_transforms,
                                     labels=True)

        train_sampler = generate_sampler(data_train)

        train_loader = DataLoader(data_train, batch_size=args.batch_size, sampler=train_sampler,
                                  num_workers=args.num_workers, pin_memory=True)
        get_masks(model, train_loader, args.fold, args.output_dir, mean_mask=True, mask_type=args.mask_type,
                  size=data_train.size, task=args.task)

    elif args.task == 'val':
        data_valid = MRIDatasetImage(args.input_dir, data_df=valid_df, preprocessing=args.preprocessing,
                                     train_transformations=train_transforms, all_transformations=all_transforms,
                                     labels=True)
        valid_loader = DataLoader(data_valid, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=True)
        get_masks(model, valid_loader, args.fold, args.output_dir, mean_mask=True,
                  mask_type=args.mask_type,
                  size=data_valid.size, task=args.task)

