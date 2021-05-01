import torch
import pandas as pd
import numpy as np
from os import path
from torch.utils.data import Dataset, sampler
import torchvision.transforms as transforms
import abc
import logging
import nibabel as nib

FILENAME_TYPE = {'full': '_T1w_space-MNI152NLin2009cSym_res-1x1x1_T1w',
                 'cropped': '_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w',
                 'skull_stripped': '_space-Ixi549Space_desc-skullstripped_T1w',
                 'gm_maps': '_T1w_segm-graymatter_space-Ixi549Space_modulated-off_probability',
                 'shepplogan': '_phantom-SheppLogan'}

class RandomNoising(object):
    """Applies a random zoom to a tensor"""

    def __init__(self, sigma=0.1):
        self.sigma = sigma

    def __call__(self, image):
        import random

        sigma = random.uniform(0, self.sigma)
        dist = torch.distributions.normal.Normal(0, sigma)
        return image + dist.sample(image.shape)


class RandomSmoothing(object):
    """Applies a random zoom to a tensor"""

    def __init__(self, sigma=1):
        self.sigma = sigma

    def __call__(self, image):
        import random
        from scipy.ndimage import gaussian_filter

        sigma = random.uniform(0, self.sigma)
        image = gaussian_filter(image, sigma)  # smoothing of data
        image = torch.from_numpy(image).float()
        return image


class RandomCropPad(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, image):
        dimensions = len(image.shape) - 1
        crop = np.random.randint(-self.length, self.length, dimensions)
        if dimensions == 2:
            output = torch.nn.functional.pad(image, (-crop[0], crop[0], -crop[1], crop[1]))
        elif dimensions == 3:
            output = torch.nn.functional.pad(image, (-crop[0], crop[0], -crop[1], crop[1], -crop[2], crop[2]))
        else:
            raise ValueError("RandomCropPad is only available for 2D or 3D data.")
        return output

class MinMaxNormalization(object):
    """Normalizes a tensor between 0 and 1"""

    def __call__(self, image):
        return (image - image.min()) / (image.max() - image.min())

class ToTensor(object):
    """Convert image type to Tensor and diagnosis to diagnosis code"""

    def __call__(self, image):
        np.nan_to_num(image, copy=False)
        image = image.astype(float)

        return torch.from_numpy(image[np.newaxis, :]).float()

def get_transforms(mode, minmaxnormalization=True, data_augmentation=None):
    """
    Outputs the transformations that will be applied to the dataset
    :param mode: (str) input used by the network. Chosen from ['image', 'patch', 'roi', 'slice'].
    :param minmaxnormalization: (bool) if True will perform MinMaxNormalization
    :param data_augmentation: (list[str]) list of data augmentation performed on the training set.
    :return:
    - container transforms.Compose including transforms to apply in train and evaluation mode.
    - container transforms.Compose including transforms to apply in evaluation mode only.
    """
    augmentation_dict = {"Noise": RandomNoising(sigma=0.1),
                         "Erasing": transforms.RandomErasing(),
                         "CropPad": RandomCropPad(10),
                         "Smoothing": RandomSmoothing(),
                         "None": None}
    if data_augmentation:
        augmentation_list = [augmentation_dict[augmentation] for augmentation in data_augmentation]
    else:
        augmentation_list = []

    if minmaxnormalization:
        transformations_list = [MinMaxNormalization()]
    else:
        transformations_list = []

    all_transformations = transforms.Compose(transformations_list)
    train_transformations = transforms.Compose(augmentation_list)

    return train_transformations, all_transformations

def load_data(tsv_path, diagnoses_list,
              split, n_splits=None, baseline=True, logger=None):
    if logger is None:
        logger = logging

    train_df = pd.DataFrame()
    valid_df = pd.DataFrame()

    if n_splits is None:
        train_path = path.join(tsv_path, 'train')
        valid_path = path.join(tsv_path, 'validation')

    else:
        train_path = path.join(tsv_path, f'train_splits-{n_splits}', f'split-{split}')
        valid_path = path.join(tsv_path, f'validation_splits-{n_splits}', f'split-{split}')

    logger.debug("Train path %s" % train_path)
    logger.debug("Valid path %s" % valid_path)

    for diagnosis in diagnoses_list:

        if baseline:
            train_diagnosis_path = path.join(
                train_path, diagnosis + '_baseline.tsv')
        else:
            train_diagnosis_path = path.join(train_path, diagnosis + '.tsv')

        valid_diagnosis_path = path.join(
            valid_path, diagnosis + '_baseline.tsv')

        train_diagnosis_df = pd.read_csv(train_diagnosis_path, sep='\t')
        valid_diagnosis_df = pd.read_csv(valid_diagnosis_path, sep='\t')

        train_df = pd.concat([train_df, train_diagnosis_df])
        valid_df = pd.concat([valid_df, valid_diagnosis_df])

    train_df.reset_index(inplace=True, drop=True)
    valid_df.reset_index(inplace=True, drop=True)

    train_df["cohort"] = "single"
    valid_df["cohort"] = "single"

    return train_df, valid_df

def get_nii_path(caps_dict, participant_id, session_id, cohort, preprocessing):

    if cohort not in caps_dict.keys():
        raise ValueError('Cohort names in labels and CAPS definitions do not match.')

    if preprocessing == "t1-linear":
        image_path = path.join(caps_dict[cohort], 'subjects', participant_id, session_id,
                               't1_linear',
                               participant_id + '_' + session_id +
                               FILENAME_TYPE['cropped'] + '.nii.gz')
    elif preprocessing == "t1-extensive":
        image_path = path.join(caps_dict[cohort], 'subjects', participant_id, session_id,
                               't1', 'spm', 'segmentation', 'normalized_space',
                               participant_id + '_' + session_id +
                               FILENAME_TYPE['skull_stripped'] + '.nii.gz')
    else:
        raise ValueError(
            "Preprocessing %s must be in ['t1-linear', 't1-extensive']." %
            preprocessing)

    return image_path

class MRIDataset(Dataset):
    """Abstract class for all derived MRIDatasets."""

    def __init__(self, caps_directory, data_file,
                 preprocessing, transformations, labels,
                 augmentation_transformations=None, multi_cohort=False):
        self.caps_dict = self.create_caps_dict(caps_directory, multi_cohort)
        self.transformations = transformations
        self.augmentation_transformations = augmentation_transformations
        self.eval_mode = False
        self.labels = labels
        self.diagnosis_code = {
            'CN': 0,
            'BV': 1,
            'AD': 1,
            'sMCI': 0,
            'pMCI': 1,
            'MCI': 1,
            'unlabeled': -1}
        self.preprocessing = preprocessing

        if not hasattr(self, 'elem_index'):
            raise ValueError(
                "Child class of MRIDataset must set elem_index attribute.")
        if not hasattr(self, 'mode'):
            raise ValueError(
                "Child class of MRIDataset must set mode attribute.")

        # Check the format of the tsv file here
        if isinstance(data_file, str):
            self.df = pd.read_csv(data_file, sep='\t')
        elif isinstance(data_file, pd.DataFrame):
            self.df = data_file
        else:
            raise Exception('The argument data_file is not of correct type.')

        if not multi_cohort:
            self.df["cohort"] = "single"

        mandatory_col = {"participant_id", "session_id"}
        if self.labels:
            mandatory_col.add("diagnosis")
        if multi_cohort:
            mandatory_col.add("cohort")
        if self.elem_index == "mixed":
            mandatory_col.add("%s_id" % self.mode)

        if not mandatory_col.issubset(set(self.df.columns.values)):
            raise Exception("the data file is not in the correct format."
                            "Columns should include %s" % mandatory_col)

        unique_diagnoses = set(self.df.diagnosis)
        unique_codes = set()
        for diagnosis in unique_diagnoses:
            unique_codes.add(self.diagnosis_code[diagnosis])
        # if len(unique_codes) == 1:
        #     warnings.warn("The diagnoses found in the DataFrame %s only corresponds to one class %s. "
        #                   "If you want to run a binary classification please change the labels involved."
        #                   % (unique_diagnoses, unique_codes))

        self.elem_per_image = self.num_elem_per_image()
        self.size = self[0]['image'].size()

    def __len__(self):
        return len(self.df) * self.elem_per_image

    @staticmethod
    def create_caps_dict(caps_directory, multi_cohort):
        caps_dict = {'single': caps_directory}
        return caps_dict

    def _get_path(self, participant, session, cohort, mode="image"):

        if cohort not in self.caps_dict.keys():
            raise ValueError('Cohort names in labels and CAPS definitions do not match.')

        image_path = path.join(self.caps_dict[cohort], 'subjects', participant, session,
                                   'deeplearning_prepare_data', '%s_based' % mode, 't1_linear',
                                   participant + '_' + session
                                   + FILENAME_TYPE['cropped'] + '.pt')

        return image_path

    def _get_meta_data(self, idx):
        image_idx = idx // self.elem_per_image
        participant = self.df.loc[image_idx, 'participant_id']
        session = self.df.loc[image_idx, 'session_id']
        cohort = self.df.loc[image_idx, 'cohort']

        if self.elem_index is None:
            elem_idx = idx % self.elem_per_image
        elif self.elem_index == "mixed":
            elem_idx = self.df.loc[image_idx, '%s_id' % self.mode]
        else:
            elem_idx = self.elem_index

        if self.labels:
            diagnosis = self.df.loc[image_idx, 'diagnosis']
            label = self.diagnosis_code[diagnosis]
        else:
            label = self.diagnosis_code['unlabeled']

        return participant, session, cohort, elem_idx, label

    def _get_full_image(self):

        participant_id = self.df.loc[0, 'participant_id']
        session_id = self.df.loc[0, 'session_id']
        cohort = self.df.loc[0, 'cohort']

        try:
            image_path = self._get_path(participant_id, session_id, cohort, mode="image")
            image = torch.load(image_path)
        except FileNotFoundError:
            image_path = get_nii_path(
                self.caps_dict,
                participant_id,
                session_id,
                cohort=cohort,
                preprocessing=self.preprocessing)
            image_nii = nib.load(image_path)
            image_np = image_nii.get_fdata()
            image = ToTensor()(image_np)

        return image

    @abc.abstractmethod
    def __getitem__(self, idx):
        pass

    @abc.abstractmethod
    def num_elem_per_image(self):
        pass

    def eval(self):
        self.eval_mode = True
        return self

    def train(self):
        self.eval_mode = False
        return self


class MRIDatasetImage(MRIDataset):
    """Dataset of MRI organized in a CAPS folder."""

    def __init__(self, caps_directory, data_df,
                 preprocessing='t1-linear', train_transformations=None,
                 labels=True, all_transformations=None, multi_cohort=False):
        """
        Args:
            caps_directory (string): Directory of all the images.
            data_file (string or DataFrame): Path to the tsv file or DataFrame containing the subject/session list.
            preprocessing (string): Defines the path to the data in CAPS.
            train_transformations (callable, optional): Optional transform to be applied only on training mode.
            labels (bool): If True the diagnosis will be extracted from the given DataFrame.
            all_transformations (callable, options): Optional transform to be applied during training and evaluation.
            multi_cohort (bool): If True caps_directory is the path to a TSV file linking cohort names and paths.
        """
        self.elem_index = None
        self.mode = "image"
        super().__init__(caps_directory, data_df, preprocessing,
                         augmentation_transformations=train_transformations, labels=labels,
                         transformations=all_transformations, multi_cohort=multi_cohort)

    def __getitem__(self, idx):
        participant, session, cohort, _, label = self._get_meta_data(idx)

        image_path = self._get_path(participant, session, cohort, "image")
        image = torch.load(image_path)

        if self.transformations:
            image = self.transformations(image)

        if self.augmentation_transformations and not self.eval_mode:
            image = self.augmentation_transformations(image)

        sample = {'image': image, 'label': label, 'participant_id': participant, 'session_id': session,
                  'image_path': image_path}

        return sample

    def num_elem_per_image(self):
        return 1


def generate_sampler(dataset, sampler_option='random'):
    """
    Returns sampler according to the wanted options
    :param dataset: (MRIDataset) the dataset to sample from
    :param sampler_option: (str) choice of sampler
    :return: (Sampler)
    """
    df = dataset.df
    # To be changed for non-binary classification
    count = np.zeros(2)

    for idx in df.index:
        label = df.loc[idx, "diagnosis"]
        key = dataset.diagnosis_code[label]
        count[key] += 1

    weight_per_class = 1 / np.array(count)
    weights = []

    for idx, label in enumerate(df["diagnosis"].values):
        key = dataset.diagnosis_code[label]
        weights += [weight_per_class[key]] * dataset.elem_per_image

    if sampler_option == 'random':
        return sampler.RandomSampler(weights)
    elif sampler_option == 'weighted':
        return sampler.WeightedRandomSampler(weights, len(weights))
    else:
        raise NotImplementedError(f"The option {sampler_option} for sampler is not implemented")
