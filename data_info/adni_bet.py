import nipype.interfaces.fsl as fsl
from nilearn.plotting import plot_anat
import nilearn.plotting
import os

PATH_TO_MRI = '/data/caps/subjects'
# sub-ADNI099S0291_ses-M00_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz
# sub-ADNI128S0528_ses-M00_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz
def adni_fsl_bet():
    i = 0
    for sub in os.listdir(PATH_TO_MRI):
        for ses in os.listdir(os.path.join(PATH_TO_MRI,sub)):

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


if __name__ == '__main__':

    adni_fsl_bet()
