# coding: utf8

"""The project is inspired by the clinica /clinicadl library, the code is taken from https://github.com/aramis-lab/AD-DL"""

import logging
import sys

LOG_LEVELS = [logging.WARNING, logging.INFO, logging.DEBUG]


class StdLevelFilter(logging.Filter):
    def __init__(self, err=False):
        super().__init__()
        self.err = err

    def filter(self, record):
        if record.levelno <= logging.INFO:
            return not self.err
        return self.err


def return_logger(verbose, name_fn):
    logger = logging.getLogger(name_fn)
    if verbose < len(LOG_LEVELS):
        logger.setLevel(LOG_LEVELS[verbose])
    else:
        logger.setLevel(logging.DEBUG)
    stdout = logging.StreamHandler(sys.stdout)
    stdout.addFilter(StdLevelFilter())
    stderr = logging.StreamHandler(sys.stderr)
    stderr.addFilter(StdLevelFilter(err=True))
    # create formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(message)s")
    # add formatter to ch
    stdout.setFormatter(formatter)
    stderr.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(stdout)
    logger.addHandler(stderr)

    return logger

class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        import numpy as np

        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')

        if mode == 'min':
            self.is_better = lambda a, best: a < best - best * min_delta
        if mode == 'max':
            self.is_better = lambda a, best: a > best + best * min_delta
def commandline_to_json(commandline, logger=None, filename="commandline.json"):
    """
    This is a function to write the python argparse object into a json file.
    This helps for DL when searching for hyperparameters
    Args:
        commandline: (Namespace or dict) the output of `parser.parse_known_args()`
        logger: (logging object) writer to stdout and stderr
        filename: (str) name of the JSON file.
    :return:
    """
    if logger is None:
        logger = logging

    import json
    import os
    from copy import copy

    if isinstance(commandline, dict):
        commandline_arg_dict = copy(commandline)
    else:
        commandline_arg_dict = copy(vars(commandline))
    output_dir = commandline_arg_dict['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # remove these entries from the commandline log file
    if 'func' in commandline_arg_dict:
        del commandline_arg_dict['func']

    if 'output_dir' in commandline_arg_dict:
        del commandline_arg_dict['output_dir']

    if 'launch_dir' in commandline_arg_dict:
        del commandline_arg_dict['launch_dir']

    if 'name' in commandline_arg_dict:
        del commandline_arg_dict['name']

    if 'verbose' in commandline_arg_dict:
        del commandline_arg_dict['verbose']

    # save to json file
    json = json.dumps(commandline_arg_dict, skipkeys=True, indent=4)
    logger.info("Path of json file: %s" % os.path.join(output_dir, "commandline.json"))
    f = open(os.path.join(output_dir, filename), "w")
    f.write(json)
    f.close()


def display_table(table_path):
    """Custom function to display the clinicadl tsvtool analysis output"""
    import pandas as pd
    from IPython.display import display

    OASIS_analysis_df = pd.read_csv(table_path, sep='\t')
    OASIS_analysis_df.set_index("diagnosis", drop=True, inplace=True)
    columns = ["n_subjects", "n_scans",
               "mean_age", "std_age", "min_age", "max_age",
               "sexF", "sexM",
               "mean_MMSE", "std_MMSE", "min_MMSE", "max_MMSE",
               "CDR_0", "CDR_0.5", "CDR_1", "CDR_2", "CDR_3"]

    # Print formatted table
    format_columns = ["subjects", "scans", "age", "sex", "MMSE", "CDR"]
    format_df = pd.DataFrame(index=OASIS_analysis_df.index, columns=format_columns)
    for idx in OASIS_analysis_df.index.values:
        row_str = "%i; %i; %.1f ± %.1f [%.1f, %.1f]; %iF / %iM; %.1f ± %.1f [%.1f, %.1f]; 0: %i, 0.5: %i, 1: %i, 2:%i, 3:%i" % tuple([OASIS_analysis_df.loc[idx, col] for col in columns])
        row_list = row_str.split(';')
        format_df.loc[idx] = row_list

    format_df.index.name = None
    display(format_df)

def visualize_image(decoder, dataloader, visualization_path, nb_images=1):
    """
    Writes the nifti files of images and their reconstructions by an autoencoder.
    Args:
        decoder: (Autoencoder) Autoencoder constructed from a CNN with the Autoencoder class.
        dataloader: (DataLoader) wrapper of the dataset.
        visualization_path: (str) directory in which the inputs and reconstructions will be stored.
        nb_images: (int) number of images to reconstruct.
    """
    import nibabel as nib
    import numpy as np
    import os
    from train_run import check_and_clean

    check_and_clean(visualization_path)

    dataset = dataloader.dataset
    decoder.eval()
    dataset.eval()

    for image_index in range(nb_images):
        data = dataset[image_index]
        image = data["image"].unsqueeze(0).cuda()
        output = decoder(image)

        output_np = output.squeeze(0).squeeze(0).cpu().detach().numpy()
        input_np = image.squeeze(0).squeeze(0).cpu().detach().numpy()
        output_nii = nib.Nifti1Image(output_np, np.eye(4))
        input_nii = nib.Nifti1Image(input_np, np.eye(4))
        nib.save(output_nii, os.path.join(
            visualization_path, 'output-%i.nii.gz' % image_index))
        nib.save(input_nii, os.path.join(
            visualization_path, 'input-%i.nii.gz' % image_index))

def plot_central_cuts(img, title="", t=None):
    """
    param image: tensor or np array of shape (CxDxHxW) if t is None
    param image: tensor or np array of shape (TxCxDxHxW) if t is not None
    """
    if t is not None:
        img = img[t]
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(3 * 6, 6))
    fig.suptitle(title)
    axes[0].imshow(img[0, img.shape[1] // 2, :, :])
    axes[1].imshow(img[0, :, img.shape[2] // 2, :])
    axes[2].imshow(img[0, :, :, img.shape[3] // 2])
    plt.show()

def display_interpretation(interp_img, data_img, cut_coords=(40, 25, 55), threshold=0.35, name = 'mean'):
    import matplotlib.pyplot as plt
    from nilearn import plotting
    from os import path
    import nibabel as nib
    import pandas as pd
    import numpy as np


    fig, axes = plt.subplots(figsize=(16, 8))
    roi_img = nib.Nifti1Image(interp_img, affine=np.eye(4))
    print(data_img.shape)
    print(np.squeeze(data_img).cpu().detach().numpy().shape)
    bim_img = nib.Nifti1Image(np.squeeze(data_img).cpu().detach().numpy(), affine=np.eye(4))
    if cut_coords is None:
        plotting.plot_roi(roi_img, bim_img, axes=axes, colorbar=True, cmap='jet',
                          threshold=threshold)
    else:
        plotting.plot_roi(roi_img, bim_img, cut_coords=cut_coords, axes=axes, colorbar=True, cmap='jet', threshold=threshold)
    plt.show()
    fig.savefig("grad_cam_{}".format(name), bbox_inches='tight')
