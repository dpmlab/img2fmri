import os
import glob
import pickle
import joblib
from pathlib import Path

import zipfile
import wget
from natsort import natsorted
from tqdm import tqdm, trange
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import PatchCollection
import pandas as pd
from sklearn.linear_model import Ridge
from scipy import stats

import torch
from torchvision import models
from torchvision import transforms
from torch.autograd import Variable

import nibabel as nib
from nipype.interfaces import afni
from nipype.interfaces import fsl

ALL_ROIS = ["EarlyVis", "OPA", "LOC", "RSC", "PPA"]


# Predicts an fMRI response for each image in input_dir, and saves it to output_dir. Predicts only
# into ROIs specified in roi_list.
def predict(input_dir, output_dir, roi_list=['LOC', 'PPA', 'RSC']):
    make_directories(input_dir=input_dir, output_dir=output_dir)
    generate_activations(input_dir)
    generate_brains(roi_list)
    transform_to_MNI()
    smooth_brains()
    average_subjects(input_dir, output_dir)


# Pushes input images through our pretrained resnet18 model and saves the activations.
# Can be easily modified to a different network or layer if desired.
def generate_activations(input_dir, output_dir=""):
    if output_dir == "": 
        output_dir = f"temp/activations/"

    # Default input image transformations for ImageNet
    scaler = transforms.Resize((224, 224))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()

    # Load our pretrained model
    model = models.resnet18(pretrained=True)
    model.eval()

    for filename in tqdm(glob.glob(f"{input_dir}/*"), desc='Pushing images through CNN'):
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


# Pushes each image activation through our regression model and saves an fMRI response in each 
# of our three training subjects' brain space.
def generate_brains(roi_list=["LOC", "RSC", "PPA"]):
    if roi_list is None: return
    num_subjects = 3
    ridge_p_grid = {'alpha': np.logspace(1, 5, 10)}
    # Load presaved array which contains the shape of each training subject's voxel ROI
    shape_array = np.load('derivatives/shape_array.npy', allow_pickle=True)

    # For each subject, for each input file, predict all ROIs for that subject and save prediction
    num_predictions = num_subjects * len(glob.glob('temp/activations/*'))
    with tqdm(total=num_predictions, desc='Predicting images into each subject\'s brain') as pbar:
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

                    # LH Nanmean for this subject's ROI
                    temp_brain = np.array([subj_brain[LH_T1_mask],
                                pred_brain[:int(shape_array[subj][ALL_ROIS.index(roi)][0])]])
                    temp_brain = np.nanmean(temp_brain, axis=0)
                    subj_brain[LH_T1_mask] = temp_brain
                    # RH Nanmean for this subject's ROI
                    temp_brain = np.array([subj_brain[RH_T1_mask],
                                pred_brain[int(shape_array[subj][ALL_ROIS.index(roi)][0]):]])
                    temp_brain = np.nanmean(temp_brain, axis=0)
                    subj_brain[RH_T1_mask] = temp_brain

                nib.save(nib.Nifti1Image(subj_brain, affine=T1_mask_nib.affine),
                        f'temp/subj_space/sub{subj+1}_{Path(filename).stem}.nii.gz')


# Transforms each subject's brain volume into MNI space for all input images
def transform_to_MNI():
    total_num_images = 3 * len(glob.glob('temp/activations/*'))
    with tqdm(total=total_num_images, desc='Transforming each brain to MNI') as pbar:
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


# Smooths the voxels in MNI space with sigma=1, or FWHM≈2.355
def smooth_brains(sig=1):
    filename = f"temp/mni/*"
    for file in tqdm(glob.glob(filename), desc='Smoothing brains'):
        stem = Path(file).stem

        out = f"temp/mni_s/{stem}.gz"
        smooth = fsl.IsotropicSmooth()
        smooth.inputs.in_file = file
        smooth.inputs.sigma = sig
        smooth.inputs.out_file = out
        smooth.run()


# Averages each of our subject's brain volumes into one volume. Does this by z-scoring each volume,
# summing those volumes, and then z-scoring the sum. Saves each averaged volume in the ouput folder. 
def average_subjects(input_dir, output_dir):
    overlap = get_subj_overlap()

    for filename in tqdm(glob.glob(f'{input_dir}/*'), desc='Averaging MNI brains'):
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

# Makes directories for storing activations, intermediate brain volumes, and output brain volumes.
def make_directories(output_dir=""):
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

# Returns the indices of an MNI brain volume comprising the logical OR of our subjects' ROIs.
# Works for all 5 ROIs, and returns a given voxel index if voxel value (post smoothing) > threshold.
def get_subj_overlap(rois=['LOC', 'PPA', 'RSC']):
    threshold = 0.15
    for roi_idx, roi in enumerate(rois):
        for subj in range(0,3):
            lh = nib.load(f'derivatives/s_bool_masks/s_sub{subj+1}_LH{roi}_MNI.nii.gz').get_fdata()
            rh = nib.load(f'derivatives/s_bool_masks/s_sub{subj+1}_RH{roi}_MNI.nii.gz').get_fdata()
            LH_mask = lh > np.max(lh) * threshold
            RH_mask = rh > np.max(rh) * threshold

            if (roi_idx == 0) and (subj == 0):
                subject_overlap = LH_mask | RH_mask
            else: 
                subject_overlap = subject_overlap | LH_mask | RH_mask
    return subject_overlap