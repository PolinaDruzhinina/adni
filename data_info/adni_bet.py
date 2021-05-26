import nipype.interfaces.fsl as fsl
from nilearn.plotting import plot_anat
import nilearn.plotting
import nibabel as nib
import os
from tqdm import tqdm
import shutil

PATH_TO_MRI = '/data/caps/subjects'
PATH_TO_MASK = '/home/ADNI/data/caps/subjects'
# sub-ADNI099S0291_ses-M00_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz
# sub-ADNI128S0528_ses-M00_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz
# sub-ADNI007S0101_ses-M00_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w_brain_mask.nii.gz
# sub-ADNI128S0528_ses-M00_T1w_space-MNI152NLin2009cSym_res-1x1x1_affine.mat
# sub-ADNI128S0528_ses-M00_T1w_space-MNI152NLin2009cSym_res-1x1x1_T1w.nii.gz
def adni_fsl_bet():
    i = 0
    cropped = False
    for sub in os.listdir(PATH_TO_MRI):
        for ses in os.listdir(os.path.join(PATH_TO_MRI,sub)):
            if cropped:
                full_path = '{}/{}/{}/t1_linear/{}_{}_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz'.format(PATH_TO_MRI, sub, ses, sub,ses)
                if os.path.exists(full_path):
                    dir_path = '/home/ADNI/{}/{}/{}/t1_linear'.format(PATH_TO_MRI,sub,ses)
                    if not os.path.exists(dir_path):
                        os.makedirs(dir_path)
                    out_path = '/home/ADNI/{}/{}/{}/t1_linear/{}_{}_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w_brain.nii.gz'.format(PATH_TO_MRI,sub, ses, sub,ses)
                    skullstrip = fsl.BET(in_file=full_path, out_file=out_path,
                            mask =True)
                    res = skullstrip.run()
                    i=+1
                    print(i)
                else:
                    print("Can't find file in {}".format(full_path))
            else:
                full_path = '{}/{}/{}/t1_linear/{}_{}_T1w_space-MNI152NLin2009cSym_res-1x1x1_T1w.nii.gz'.format(
                    PATH_TO_MRI, sub, ses, sub, ses)
                if os.path.exists(full_path):
                    dir_path = '/home/ADNI/{}/{}/{}/t1_linear'.format(PATH_TO_MRI, sub, ses)
                    if not os.path.exists(dir_path):
                        os.makedirs(dir_path)
                    out_path = '/home/ADNI/{}/{}/{}/t1_linear/{}_{}_T1w_space-MNI152NLin2009cSym_res-1x1x1_T1w_brain.nii.gz'.format(
                        PATH_TO_MRI, sub, ses, sub, ses)
                    skullstrip = fsl.BET(in_file=full_path, out_file=out_path,
                                         mask=True)
                    res = skullstrip.run()
                    i +=1
                    print(i)
                else:
                    print("Can't find file in {}".format(full_path))

def copy_mat():
    for i, sub in tqdm(enumerate(os.listdir(PATH_TO_MRI))):
        for ses in os.listdir(os.path.join(PATH_TO_MRI, sub)):
            full_path = '{}/{}/{}/t1_linear/{}_{}_T1w_space-MNI152NLin2009cSym_res-1x1x1_affine.mat'.format(
                PATH_TO_MRI, sub, ses, sub, ses)
            if os.path.exists(full_path):
                shutil.copyfile(full_path, '{}/{}/{}/t1_linear/{}_{}_T1w_space-MNI152NLin2009cSym_res-1x1x1_affine.mat'.format(
                PATH_TO_MASK, sub, ses, sub, ses))

def get_brain_from_mask():
    cropped = True
    for i, sub in tqdm(enumerate(os.listdir(PATH_TO_MRI))):
        for ses in os.listdir(os.path.join(PATH_TO_MRI,sub)):
            if cropped:
                full_path = '{}/{}/{}/t1_linear/{}_{}_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz'.format(
                    PATH_TO_MRI, sub, ses, sub, ses)
                if os.path.exists(full_path):
                    mask_path = '{}/{}/{}/t1_linear/{}_{}_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w_brain_mask.nii.gz'.format(PATH_TO_MASK, sub, ses, sub, ses)
                    img = nib.load(full_path).get_data()
                    full_mask = nib.load(mask_path).get_data()
                    masked_image = img.copy()
                    masked_image[full_mask == 0] = 0
                    nib.save(nib.Nifti1Image(masked_image, affine=None), '{}/{}/{}/t1_linear/{}_{}_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz'.format(PATH_TO_MASK, sub,ses, sub, ses))
            else:
                full_path = '{}/{}/{}/t1_linear/{}_{}_T1w_space-MNI152NLin2009cSym_res-1x1x1_T1w.nii.gz'.format(
                    PATH_TO_MRI, sub, ses, sub, ses)
                if os.path.exists(full_path):
                    mask_path = '{}/{}/{}/t1_linear/{}_{}_T1w_space-MNI152NLin2009cSym_res-1x1x1_T1w_brain_mask.nii.gz'.format(
                        PATH_TO_MASK, sub, ses, sub, ses)
                    img = nib.load(full_path).get_data()
                    full_mask = nib.load(mask_path).get_data()
                    masked_image = img.copy()
                    masked_image[full_mask == 0] = 0
                    nib.save(nib.Nifti1Image(masked_image, affine=None),
                             '{}/{}/{}/t1_linear/{}_{}_T1w_space-MNI152NLin2009cSym_res-1x1x1_T1w.nii.gz'.format(
                                 PATH_TO_MASK, sub,ses, sub, ses))

if __name__ == '__main__':

    adni_fsl_bet()
   # get_brain_from_mask()
