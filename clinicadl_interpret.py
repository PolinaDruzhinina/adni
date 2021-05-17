"""The project is inspired by the clinica /clinicadl library, the code is taken from https://github.com/aramis-lab/AD-DL"""

import os
from os import path
import sys
import torch
import argparse
import pandas as pd
import numpy as np
import nibabel as nib
from torch.utils.data import DataLoader

from model import AutoEncoder, Conv5_FC3,load_model
from data import get_transforms, MRIDatasetImage
from utils import return_logger, commandline_to_json, read_json
from test import test
from Interpretation.clinica_dl_vanilla_bp import VanillaBackProp

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', help='path to output dir', type=str, default='/home/ADNI/image_cnn_pmci_cn')
parser.add_argument('--input_dir', help='path to input dir', type=str, default='/data/caps')
parser.add_argument('--tsv_path', help='path', type=str, default='/home/ADNI/data_info/labels_pamci_lists/test')
parser.add_argument("--diagnoses", help="Labels that must be extracted from merged_tsv.",
                    nargs="+", type=str, choices=['AD', 'BV', 'CN', 'MCI', 'sMCI', 'pMCI'], default=['pMCI'])
parser.add_argument("--target_diagnosis", help="Labels that must be extracted from merged_tsv.",
                    nargs="+", type=str, choices=['AD', 'BV', 'CN', 'MCI', 'sMCI', 'pMCI'], default='CN')
parser.add_argument('--keep_true', default=None, type=bool, help='keep_true')
parser.add_argument("--vmax", type=float, default=0.5, help="Maximum value used in 2D image display.")
parser.add_argument(
        "--nifti_template_path", type=str, default=None,
        help="Path to a nifti template to retrieve affine values.")
parser.add_argument('--name', help='name', type=str, default='group-pMCI_target-CN')
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
parser.add_argument('--dropout', default=0.5, type=float, help='initial dropout')
parser.add_argument('--mode', type=str, choices=['image', 'slice'], default='image')
parser.add_argument('--model', help='model', type=str, default='Conv5_FC3')
parser.add_argument('--minmaxnormalization', default=True, help='MinMaxNormalization')
parser.add_argument('--data_augmentation', default=None, help='Augmentation')
parser.add_argument('--verbose', '-v', action='count', default=0)
parser.add_argument('--selection', type=str, default=['best_loss', 'best_balanced_accuracy'])
args = parser.parse_args()

sys.stdout.flush()




def individual_backprop(options):

    main_logger = return_logger(options.verbose, "main process")

    fold_list = [fold for fold in os.listdir(options.output_dir) if fold[:5:] == "fold-"]
    if len(fold_list) == 0:
        raise ValueError("No folds were found at path %s" % options.output_dir)

    if os.path.exists(path.join(options.output_dir, 'commandline.json')):
        model_options = argparse.Namespace()
        model_options = read_json(model_options, path.join(options.output_dir, 'commandline.json'))
        model_options.gpu = options.gpu


    if options.tsv_path is None:
        options.tsv_path = model_options.tsv_path
    if options.input_dir is None:
        options.input_dir = model_options.input_dir
    if options.target_diagnosis is None:
        options.target_diagnosis = options.diagnosis

    for fold in fold_list:
        main_logger.info(fold)
        for selection in options.selection:
            results_path = path.join(options.output_dir, fold, 'gradients',
                                     selection, options.name)

            criterion = torch.nn.CrossEntropyLoss(reduction="sum")

            # Data management (remove data not well predicted by the CNN)
            test_df = pd.DataFrame()
            main_logger.debug("Test path %s" % options.tsv_path)
            print(options.diagnoses)
            for diagnosis in options.diagnoses:
                print(diagnosis)
                test_diagnosis_path = path.join(
                        options.tsv_path, diagnosis + '_baseline.tsv')

                test_diagnosis_df = pd.read_csv(test_diagnosis_path, sep='\t')
                test_df = pd.concat([test_df, test_diagnosis_df])

                test_df.reset_index(inplace=True, drop=True)
                test_df["cohort"] = "single"

            test_df.reset_index(drop=True, inplace=True)

            # Model creation
            _, all_transforms = get_transforms(options.mode,
                                               minmaxnormalization=options.minmaxnormalization)

            data_example = MRIDatasetImage(options.input_dir, data_df=test_df, preprocessing=options.preprocessing,
                                        train_transformations=None, all_transformations=all_transforms,
                                        labels=True)

            if options.id_gpu is not None:
                torch.cuda.set_device(options.id_gpu)

            model = eval(options.model)(dropout=options.dropout)
            if options.gpu:
                model.cuda()
            else:
                model.cpu()
            
            model_dir = os.path.join(options.output_dir, fold, 'models', selection)
            model, best_epoch = load_model(model, model_dir, gpu=options.gpu, filename='model_best.pth.tar')
            options.output_dir = results_path
            commandline_to_json(options, logger=main_logger)

            # Keep only subjects who were correctly / wrongly predicted by the network
            if options.keep_true is not None:
                dataloader = DataLoader(data_example,
                                        batch_size=options.batch_size,
                                        shuffle=False,
                                        num_workers=options.num_workers,
                                        pin_memory=True)

                results_df, _ = test(model, dataloader, options.gpu, criterion, options.mode, use_labels=True)

                sorted_df = test_df.sort_values(['participant_id', 'session_id']).reset_index(drop=True)
                results_df = results_df.sort_values(['participant_id', 'session_id']).reset_index(drop=True)

                if options.keep_true:
                    test_df = sorted_df[results_df.true_label == results_df.predicted_label].reset_index(drop=True)
                else:
                    test_df = sorted_df[results_df.true_label != results_df.predicted_label].reset_index(drop=True)

            if len(test_df) > 0:

                # Save the tsv files used for the saliency maps
                test_df.to_csv(path.join('data.tsv'), sep='\t', index=False)

                dataset = MRIDatasetImage(options.input_dir, data_df=test_df, preprocessing=options.preprocessing,
                                               train_transformations=None, all_transformations=all_transforms,
                                               labels=True)
                train_loader = DataLoader(dataset,
                                          batch_size=options.batch_size,
                                          shuffle=True,
                                          num_workers=options.num_workers,
                                          pin_memory=True)

                interpreter = VanillaBackProp(model, gpu=options.gpu)

                for data in train_loader:
                    if options.gpu:
                        input_batch = data['image'].cuda()
                    else:
                        input_batch = data['image']

                    map_np = interpreter.generate_gradients(input_batch,
                                                            dataset.diagnosis_code[options.target_diagnosis])
                    for i in range(options.batch_size):
                        single_path = path.join(results_path, data['participant_id'][i], data['session_id'][i])
                        os.makedirs(single_path, exist_ok=True)

                        if len(dataset.size) == 4:
                            if options.nifti_template_path is not None:
                                image_nii = nib.load(options.nifti_template_path)
                                affine = image_nii.affine
                            else:
                                affine = np.eye(4)

                            map_nii = nib.Nifti1Image(map_np[i, 0, :, :, :], affine)
                            nib.save(map_nii, path.join(single_path, "map.nii.gz"))
                        else:
                            jpg_path = path.join(single_path, "map.jpg")
                            plt.imshow(map_np[i, 0, :, :], cmap="coolwarm", vmin=-options.vmax, vmax=options.vmax)
                            plt.colorbar()
                            plt.savefig(jpg_path)
                            plt.close()
                        np.save(path.join(single_path, "map.npy"), map_np[i])

if __name__ == '__main__':
    args = parser.parse_args()

    individual_backprop(args)
