"""
imgtofmri python package for predicting group-level fMRI responses to individual images.
Tutorial and analysis found at: https://github.com/dpmlab/imgtofmri
Author: Maxwell Bennett mbb2176@columbia.edu
"""
import os
import glob
import argparse
import joblib
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import numpy as np
from scipy import stats

import torch
from torch.autograd import Variable
from torchvision import models
from torchvision import transforms

import nibabel as nib
from nipype.interfaces import afni
from nipype.interfaces import fsl

from utils import get_subj_overlap

# Predictions can be made for these ROIs, however default is ['LOC', 'PPA', 'RSC'] given signif.
ALL_ROIS = ["EarlyVis", "OPA", "LOC", "RSC", "PPA"]
TQDM_FORMAT = '{l_bar}{bar:10}{r_bar}{bar:-10b}' # default progress bar format


def main():
    """
    Command line interface version to predict group-level fMRI responses to individual image frames
    usage: python imgtofmri.py [-h] --input input_dir [--output output_dir] [--rois each roi here]
                               [--sigma sigma_val] [--center_crop center_crop]
    """
    args = get_args()
    make_directories(output_dir=args.output)
    predict(args.input[0], args.output, args.rois, args.sigma, args.center_crop)


def predict(input_dir, output_dir, roi_list=['LOC', 'PPA', 'RSC'], sigma=1, center_crop=False):
    """
    Predicts an fMRI response for each image in input_dir, and saves it to output_dir.
    Predicts only to ROIs specified in roi_list. sigma specifies smoothing constant.
    """
    make_directories(output_dir=output_dir)
    generate_activations(input_dir, center_crop=center_crop)
    generate_brains(roi_list)
    transform_to_MNI()
    smooth_brains(sigma)
    average_subjects(input_dir, output_dir)


def generate_activations(input_dir, output_dir="", center_crop=False):
    """
    Pushes input images through our pretrained resnet18 model and saves the activations.
    Can be easily modified to a different network or layer if desired by changing model.
    center_crop -- specifies whether each image is resized or 'squished' to a square, vs. cropped.
    """
    if output_dir == "":
        output_dir = "temp/activations/"

    # Default input image transformations for ImageNet
    if center_crop:
        scaler = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop((224, 224))
        ])
    else:
        scaler = transforms.Resize((224, 224))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()

    # Load our pretrained model
    model = models.resnet18(weights='DEFAULT')
    model.eval()

    desc = 'Pushing images through CNN'
    for filename in tqdm(glob.glob(f"{input_dir}/*"), bar_format=TQDM_FORMAT, desc=desc):
        if Path(filename).suffix not in [".jpg", '.JPG', '.jpeg', '.JPEG', ".png", '.PNG']:
            continue

        img = Image.open(filename)
        t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))

        # Create network up to last layer and push image through
        layer_extractor = torch.nn.Sequential(*list(model.children())[:-1])
        feature_vec = layer_extractor(t_img).data.numpy().squeeze()
        feature_vec = feature_vec.flatten()

        # Save image activations
        image_name = Path(filename).stem
        np.save(f"{output_dir}/{image_name}.npy", feature_vec)
        img.close()


def generate_brains(roi_list=["LOC", "RSC", "PPA"]):
    """
    Pushes each image activation through our regression model and saves an fMRI response in each
    of our three training subjects' brain space.
    """
    if roi_list is None or len(roi_list) == 0: return
    num_subjects = 3
    # Load presaved array which contains the shape of each training subject's voxel ROI
    shape_array = np.load('derivatives/shape_array.npy', allow_pickle=True)

    # For each subject, for each input file, predict all ROIs for that subject and save prediction
    num_predictions = num_subjects * len(glob.glob('temp/activations/*'))
    desc = 'Predicting images into each subject\'s brain'
    with tqdm(total=num_predictions, bar_format=TQDM_FORMAT, desc=desc) as pbar:
        for subj in range(num_subjects):
            for filename in glob.glob('temp/activations/*'):
                pbar.update(1)
                actv  = np.load(open(filename, 'rb'), allow_pickle=True)

                for roi_idx, roi in enumerate(roi_list):
                    model = joblib.load(f'models/subj{subj+1}_{roi}_model.pkl')
                    y_pred_brain = model.predict([actv])
                    pred_brain = y_pred_brain[0]

                    # Get left hemisphere voxel mask
                    T1_mask_nib = nib.load(f"derivatives/bool_masks/derivatives_spm_sub-"\
                                        f"CSI{subj+1}_sub-CSI{subj+1}_mask-LH{roi}.nii.gz")
                    T1_mask_shape = T1_mask_nib.header.get_data_shape()[0:3]
                    LH_T1_mask = T1_mask_nib.get_fdata() > 0  # 3D boolean array
                    # Get right hemisphere voxel mask
                    T1_mask_nib = nib.load(f"derivatives/bool_masks/derivatives_spm_sub-"\
                                        f"CSI{subj+1}_sub-CSI{subj+1}_mask-RH{roi}.nii.gz")
                    RH_T1_mask = T1_mask_nib.get_fdata() > 0  # 3D boolean array

                    # Initialize subj's 3d volume if first time through
                    if roi_idx == 0:
                        subj_brain = np.empty(T1_mask_shape)
                        subj_brain[:, :, :] = np.NaN

                    # Left hemisphere nanmean for this subject's ROI
                    temp_brain = np.array([subj_brain[LH_T1_mask],
                                pred_brain[:int(shape_array[subj][ALL_ROIS.index(roi)][0])]])
                    temp_brain = np.nanmean(temp_brain, axis=0)
                    subj_brain[LH_T1_mask] = temp_brain
                    # Right hemisphere nanmean for this subject's ROI
                    temp_brain = np.array([subj_brain[RH_T1_mask],
                                pred_brain[int(shape_array[subj][ALL_ROIS.index(roi)][0]):]])
                    temp_brain = np.nanmean(temp_brain, axis=0)
                    subj_brain[RH_T1_mask] = temp_brain

                nib.save(nib.Nifti1Image(subj_brain, affine=T1_mask_nib.affine),
                        f'temp/subj_space/sub{subj+1}_{Path(filename).stem}.nii.gz')


def transform_to_MNI():
    """ Transforms each subject's brain volume into MNI space for all input images """
    total_num_images = 3 * len(glob.glob('temp/activations/*'))
    desc = 'Transforming each brain to MNI'
    with tqdm(total=total_num_images, bar_format=TQDM_FORMAT, desc=desc) as pbar:
        for subj in range(1,4):
            filename = f'temp/subj_space/sub{subj}*'
            for file in glob.glob(filename):
                pbar.update(1)
                stem = Path(file).stem
                resample = afni.Resample()
                resample.inputs.in_file = file
                resample.inputs.master = f"derivatives/T1/sub-CSI{subj}_ses-16_anat_sub-CSI{subj}" \
                                        f"_ses-16_T1w.nii.gz"
                resample.inputs.out_file = f'temp/temp/{stem}'
                resample.run()

                aw = fsl.ApplyWarp()
                aw.inputs.in_file = f'temp/temp/{stem}'
                aw.inputs.ref_file = f'derivatives/sub{subj}.anat/T1_to_MNI_nonlin.nii.gz'
                aw.inputs.field_file = f'derivatives/sub{subj}.anat/T1_to_MNI_nonlin_coeff.nii.gz'
                aw.inputs.premat = f'derivatives/sub{subj}.anat/T1_nonroi2roi.mat'
                aw.inputs.interp = 'nn'
                aw.inputs.out_file = f'temp/mni/{stem}.gz' # Note: stem here contains '*.nii'
                aw.run()
                os.remove(f'temp/temp/{stem}')


def smooth_brains(sig=1):
    """ Smooths the voxels in MNI space with sigma=1, or FWHM≈2.355 """
    filename = f"temp/mni/*"
    for file in tqdm(glob.glob(filename), bar_format=TQDM_FORMAT, desc='Smoothing brains'):
        stem = Path(file).stem
        out = f"temp/mni_s/{stem}.gz"
        smooth = fsl.IsotropicSmooth()
        smooth.inputs.in_file = file
        smooth.inputs.sigma = sig
        smooth.inputs.out_file = out
        smooth.run()


def average_subjects(input_dir, output_dir):
    """
    Averages each BOLD5000 subject's brain volumes into one volume by z-scoring each volume,
    summing those volumes, and then z-scoring the sum. Saves averaged volumes in the output folder.
    """
    overlap = get_subj_overlap()

    desc = 'Averaging MNI brains'
    for filename in tqdm(glob.glob(f'{input_dir}/*'), bar_format=TQDM_FORMAT, desc=desc):
        if Path(filename).suffix not in [".jpg", '.JPG', '.jpeg', '.JPEG', ".png", '.PNG']:
            continue

        stem = Path(filename).stem
        subj_mask_nib = nib.load(f'temp/mni_s/sub1_{stem}.nii.gz')
        im_brain = np.full((subj_mask_nib.shape), np.nan)
        zscored_sum = np.zeros((np.count_nonzero(overlap)))

        for subj in range(1,4):
            filename = f'temp/mni_s/sub{subj}_{stem}.nii.gz'
            brain = nib.load(filename).get_fdata()
            zscored_overlap = stats.zscore(brain[overlap])
            zscored_sum += zscored_overlap

        group_avg = stats.zscore(zscored_sum)
        im_brain[overlap] = group_avg

        nib.save(nib.Nifti1Image(im_brain, affine=subj_mask_nib.affine),
                                 f'{output_dir}/{stem}.nii.gz')


def make_directories(output_dir=""):
    """ Makes directories for activations, intermediate brain volumes, and output brain volumes. """
    temp_dirs = ['temp/activations', 'temp/subj_space',
                 'temp/temp', 'temp/mni', 'temp/mni_s',]
    for directory in temp_dirs:
        try:
            os.makedirs(directory)
        except FileExistsError:
            for f in glob.glob(f"{directory}/*"):
                os.remove(f)

    # Ensure output directory exists
    if output_dir != "":
        try:
            os.makedirs(output_dir)
        except FileExistsError:
            pass


def get_args():
    """ Defines arguments for command line interface version of prediction. """
    parser = argparse.ArgumentParser(description='Convert images into MNI brains.')
    parser.add_argument('--input', metavar='input_dir', type=dir_path, required=True, nargs=1,
                        help='input directory which contains all images to be processed.')
    parser.add_argument('--output', metavar='output_dir', type=str, default='fmri_output',
                        help='output directory where fMRI volumes will be saved.')
    parser.add_argument('--rois', metavar='rois', type=str, nargs='+',
                        default=['LOC', 'PPA', 'RSC'],
                        help='fMRI ROIs to predict, default: LOC, PPA, RSC')
    parser.add_argument('--sigma', metavar='sigma', type=float, default=1.0,
                        help='sigma for smoothing MNI brains, default=1.0')
    parser.add_argument('--center_crop', metavar='center_crop', type=int, default=0,
                        help='whether to center crop input images instead of resizing full image '\
                             ' to square. Argument passed as `"--center_crop=1`", default=0')
    return parser.parse_args()


def dir_path(string):
    """ Defining a type for rejecting non-directory inputs to command line use"""
    if os.path.isdir(string):
        return Path(string).resolve()
    raise NotADirectoryError(string)


if __name__ == '__main__':
    main()
