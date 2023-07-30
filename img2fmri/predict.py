"""
img2fmri python package for predicting group-level fMRI responses to individual images.
Tutorial and analysis found at: https://github.com/dpmlab/img2fmri
Author: Maxwell Bennett - mbb2176@columbia.edu
"""
import os
from os.path import join
import glob
import argparse
import joblib
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import numpy as np
from scipy import stats
from sklearn.exceptions import InconsistentVersionWarning
import warnings

import torch
from torch.autograd import Variable
from torchvision import models
from torchvision import transforms

import nibabel as nib
from nipype.interfaces import afni
from nipype.interfaces import fsl

from img2fmri.utils import get_subj_overlap, extract_frames, load_frames

# Predictions can be made for these ROIs. Defaults to ['LOC', 'PPA', 'RSC'].
ALL_ROIS = ['EarlyVis', 'OPA', 'LOC', 'RSC', 'PPA']
TQDM_FORMAT = '{l_bar}{bar:10}| {n}/{total_fmt} [{elapsed}<{remaining}]'


def predict(input_data, 
            output_dir='.',
            roi_list=['LOC', 'RSC', 'PPA'],
            predict_movie=False,
            sigma=1,
            center_crop=False):
    """Predicts an fMRI response for each input image, and saves it to output.
    
    Predictions for each image are first made for specified ROIs of each subject's brain, then
    transformed to MNI space, smoothed, averaged together to form a group-average, and then saved
    in the output directory.

    Parameters
    ----------
    input_data : str
        Path to an input directory of images or to a movie file to be predicted.
        If supplying a movie file, 'predict_movie' must be True.
    output_dir : str
        Relative path to directory where predicted responses are saved.
        Defaults to '.', which leads to saving in current directory.
    predict_movie : bool
        Determines whether to predict response to a movie and save as a
        4D brain volume of shape (x,y,z,movie_frames). Defaults to False.
    roi_list : list
        List of voxel ROIs to predict. Defaults to ['LOC', 'PPA', 'RSC'].
    sigma : int
        Standard dev of smoothing filter used for ROIs. FWHM≈2.355*sigma.
        Defaults to 1.
    center_crop : bool
        If True, crops each image to a square, from the center of the
        image, before processing. If False, images are resized to a square before processing.
        Defaults to False.

    Raises
    ------
    NotADirectoryError
        Raises an error if a directory is not supplied as input, if not
        predicting a movie.
    """
    # Make or clear temporary directories, and ensure output directory exists
    _make_directories(output_dir=output_dir)
    # Ensure AFNI and FSL properly installed before running full pipeline
    _check_AFNI_and_FSL()

    if predict_movie:
        extract_frames(input_file=input_data, output_dir=join('temp','movie_frames'))
        generate_activations(input_dir=join('temp','movie_frames'), center_crop=center_crop)
        generate_predictions(roi_list=roi_list)
        transform_to_MNI()
        smooth_brains(sigma)
        average_subjects(image_dir=join('temp','movie_frames'),
                         output_dir=join('temp','movie_frame_predictions'),
                         roi_list=roi_list)
        movie_prediction = load_frames(input_dir=join('temp','movie_frame_predictions'))
        
        out_filename = join(output_dir, f'{Path(input_data).stem}.nii.gz')
        nib.save(nib.Nifti1Image(movie_prediction,
                                 affine=nib.load(join(os.path.dirname(__file__),
                                                              'derivatives',
                                                              'init_brain.nii.gz')).affine),
                 out_filename)
        print(f'Saved {out_filename}')
    
    else:
        # Ensure that input is a directory to predict from, if not a movie
        if not os.path.isdir(input_data):
            raise NotADirectoryError(input_data)

        generate_activations(input_dir=input_data, center_crop=center_crop)
        generate_predictions(roi_list=roi_list)
        transform_to_MNI()
        smooth_brains(sigma)
        average_subjects(image_dir=input_data, output_dir=output_dir, roi_list=roi_list)


def generate_activations(input_dir, output_dir=join('temp','activations'), center_crop=False):
    """Pushes input images through our pretrained resnet18 model and saves the activations.
    Can be modified to use a different network or layer if desired, by changing 'model'.

    Parameters
    ----------
    input_dir : str
        Relative path to input directory of images to be predicted.
    output_dir : str
        Relative path to save intermediate files used in prediction.
        Defaults to 'temp/activations/'.
    center_crop : bool
        If True, crops each image to a square, from the center of the
        image, before processing. If False, images are resized to a square before processing.
        Defaults to False.
    """
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
    for filename in tqdm(glob.glob(join(input_dir,'*')), bar_format=TQDM_FORMAT, desc=desc):
        if Path(filename).suffix not in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']:
            continue

        img = Image.open(filename)
        t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))

        # Create network up to last layer and push image through
        layer_extractor = torch.nn.Sequential(*list(model.children())[:-1])
        feature_vec = layer_extractor(t_img).data.numpy().squeeze()
        feature_vec = feature_vec.flatten()

        # Save image activations
        image_name = Path(filename).stem
        np.save(join(output_dir,f'{image_name}.npy'), feature_vec)
        img.close()


def generate_predictions(roi_list=['LOC', 'RSC', 'PPA'], 
                         input_dir=join('temp','activations'), 
                         output_dir=join('temp','subj_space')):
    """Pushes each image activation through our regression model and saves an fMRI response in each
    of our three training subjects' brain space.

    Parameters
    ----------
    roi_list : list
        List of ROIs to predict voxel responses for.
        Defaults to ['LOC', 'RSC', 'PPA'].
    input_dir : str
        Relative path to input directory of activations to predict from.
        Defaults to 'temp/activations'.
    output_dir : str
        Relative path to save predicted subj space brain responses.
        Defaults to 'temp/subj_space'.
    """
    if roi_list is None or len(roi_list) == 0: return
    num_subjects = 3
    # Load presaved array which contains the shape of each training subject's voxel ROI
    shape_array = np.load(join(os.path.dirname(__file__),'derivatives','shape_array.npy'),
                          allow_pickle=True)

    # For each subject, for each input file, predict all ROIs for that subject and save prediction
    num_predictions = num_subjects * len(glob.glob(join('temp','activations','*')))
    desc = 'Predicting images into each subject\'s brain'
    with tqdm(total=num_predictions, desc=desc, bar_format=TQDM_FORMAT, ncols=100) as pbar:
        for subj in range(num_subjects):
            for filename in glob.glob(join(input_dir,'*')):
                pbar.update(1)
                actv  = np.load(open(filename, 'rb'), allow_pickle=True)

                for roi_idx, roi in enumerate(roi_list):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", InconsistentVersionWarning)
                        model = joblib.load(join(os.path.dirname(__file__),
                                                'models',
                                                f'subj{subj+1}_{roi}_model.pkl'))
                        y_pred_brain = model.predict([actv])
                        pred_brain = y_pred_brain[0]

                    # Get left hemisphere voxel mask
                    T1_mask_file = join(os.path.dirname(__file__),
                                        'derivatives',
                                        'bool_masks',
                                        f'derivatives_spm_sub-CSI{subj+1}_sub-CSI{subj+1}_mask-'\
                                        f'LH{roi}.nii.gz')
                    T1_mask_nib = nib.load(T1_mask_file)
                    T1_mask_shape = T1_mask_nib.header.get_data_shape()[0:3]
                    LH_T1_mask = T1_mask_nib.get_fdata() > 0  # 3D boolean array
                    # Get right hemisphere voxel mask
                    T1_mask_file = join(os.path.dirname(__file__),
                                        'derivatives',
                                        'bool_masks',
                                        f'derivatives_spm_sub-CSI{subj+1}_sub-CSI{subj+1}_mask-'\
                                        f'RH{roi}.nii.gz')
                    T1_mask_nib = nib.load(T1_mask_file)
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
                
                # subj_brain = np.nan_to_num(subj_brain) # eliminate np.NaN values for compatibility
                nib.save(nib.Nifti1Image(subj_brain, affine=T1_mask_nib.affine),
                         join(output_dir,f'sub{subj+1}_{Path(filename).stem}.nii.gz'))


def transform_to_MNI(input_dir=join('temp','subj_space'), output_dir=join('temp','mni')):
    """Transforms each subject's brain volume into MNI space for all input images.

    Parameters
    ----------
    input_dir : str
        Path to input directory of responses to transform into MNI space.
        Defaults to 'temp/subj_space'.
    output_dir : str
        Relative path to save predicted responses in mni space.
        Defaults to 'temp/mni'.
    """
    total_num_images = 3 * len(glob.glob(join('temp','activations','*')))
    desc = 'Transforming each brain to MNI'
    with tqdm(total=total_num_images, bar_format=TQDM_FORMAT, desc=desc, ncols=100) as pbar:
        for s in range(1,4):
            filename = join(input_dir,f'sub{s}*')
            for file in glob.glob(filename):
                pbar.update(1)
                stem = Path(file).stem
                resample = afni.Resample()
                resample.inputs.in_file = file
                resample.inputs.master = join(os.path.dirname(__file__),'derivatives','T1', f'sub-'\
                                              f'CSI{s}_ses-16_anat_sub-CSI{s}_ses-16_T1w.nii.gz')
                resample.inputs.out_file = join('temp','resampled',f'{stem}.gz') # stem has '*.nii'
                resample.inputs.args = '-DAFNI_NIFTI_TYPE_WARN=NO' 
                resample.run()
                
                aw = fsl.ApplyWarp()
                aw.inputs.in_file = join('temp','resampled',f'{stem}.gz')
                aw.inputs.ref_file = join(os.path.dirname(__file__),'derivatives',f'sub{s}.anat',
                                          'T1_to_MNI_nonlin.nii.gz')
                aw.inputs.field_file = join(os.path.dirname(__file__),'derivatives',f'sub{s}.anat',
                                            'T1_to_MNI_nonlin_coeff.nii.gz')
                aw.inputs.premat = join(os.path.dirname(__file__),'derivatives',f'sub{s}.anat',
                                        'T1_nonroi2roi.mat')
                aw.inputs.interp = 'nn'
                aw.inputs.out_file = join(output_dir,f'{stem}.gz')
                aw.run()


def smooth_brains(sigma=1, input_dir=join('temp','mni'), output_dir=join('temp','mni_s')):
    """Smooths the voxels in MNI space.

    Parameters
    ----------
    sigma : int
        Standard deviation of smoothing filter used for each ROI.
        Defaults to 1. FWHM≈2.355*sigma.
    input_dir : str
        Path to input directory of mni space responses to smooth.
        Defaults to 'temp/mni'.
    output_dir : str
        Path to save smoothed mni space responses to.
        Defaults to 'temp/mni_s'.
    """
    filename = join(input_dir,'*')
    for file in tqdm(glob.glob(filename), bar_format=TQDM_FORMAT, desc='Smoothing brains'):
        stem = Path(file).stem
        out = join(output_dir,f'{stem}.gz')
        smooth = fsl.IsotropicSmooth()
        smooth.inputs.in_file = file
        smooth.inputs.sigma = sigma
        smooth.inputs.out_file = out
        smooth.run()


def average_subjects(image_dir,
                     input_dir=join('temp','mni_s'),
                     output_dir='.', 
                     roi_list=['LOC', 'RSC', 'PPA']):
    """Averages each BOLD5000 subject's brain volume in MNI space into one volume.
    Averages by z-scoring each volume, summing those volumes, and then z-scoring the sum.
    Saves averaged volumes in the output directory.

    Parameters
    ----------
    image_dir : str
        Path to directory of images, with filenames to be used for output.
    input_dir : str
        Path to directory of subjects' predicted responses to average.
        Defaults to 'temp/mni_s'.
    output_dir : str
        Path to output directory where group-avg responses are saved.
        Defaults to '.', which saves the files in the current directory.
    roi_list : list
        List of voxel ROIs to predict. Defaults to ['LOC', 'PPA', 'RSC'].
    """
    init_brain_nib = nib.load(join(os.path.dirname(__file__),'derivatives','s_bool_masks',
                                   's_sub1_LHOPA_MNI.nii.gz'))
    init_brain = init_brain_nib.get_fdata()

    # Calculate overlap:
    for filename in glob.glob(join(image_dir,'*')):
        if Path(filename).suffix not in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']:
            continue
        stem = Path(filename).stem
        overlap = np.full(init_brain_nib.shape, False)
        for subj in range(1,4):
            filename = join(input_dir,f'sub{subj}_{stem}.nii.gz')
            brain = nib.load(filename).get_fdata()
            overlap[np.where(brain != 0)] = True
        break
    
    desc = 'Averaging MNI brains'
    for filename in tqdm(glob.glob(join(image_dir,'*')), bar_format=TQDM_FORMAT, desc=desc):
        if Path(filename).suffix not in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']:
            continue

        stem = Path(filename).stem
        zscored_sum = np.zeros((np.count_nonzero(overlap)))
        for subj in range(1,4):
            filename = join(input_dir,f'sub{subj}_{stem}.nii.gz')
            brain = nib.load(filename).get_fdata()
            zscored_overlap = stats.zscore(brain[overlap])
            zscored_sum += zscored_overlap

        group_avg = stats.zscore(zscored_sum)
        init_brain[overlap] = group_avg
        init_brain[~overlap] = 0.0

        nib.save(nib.Nifti1Image(init_brain, 
                                 affine=init_brain_nib.affine,
                                 header=init_brain_nib.header),
                join(output_dir,f'{stem}.nii.gz'))


def _make_directories(output_dir='.'):
    """Makes directories for activations, intermediate brain volumes, and output brain volumes.

    Parameters
    ----------
    output_dir : str
        If specified, creates an directory for output if does not exist.
        Defaults to '.', or the current directory.
    """
    temp_dirs = ['activations', 'subj_space', 'resampled', 'mni', 'mni_s',
                 'movie_frames', 'movie_frame_predictions']
    for directory in temp_dirs:
        try:
            os.makedirs(join('temp',directory))
        except FileExistsError:
            for f in glob.glob(join('temp',directory,'*')):
                os.remove(f)

    # Ensure output directory exists
    if output_dir != '.':
        try:
            os.makedirs(output_dir)
        except FileExistsError:
            pass

def _check_AFNI_and_FSL():
    """Checks whether AFNI and FSL are properly installed for resampling and MNI transformation.
    Called when full predict() pipeline is called.
    """
    try:
        resample = afni.Resample()
        resample.inputs.in_file = join(os.path.dirname(__file__),'derivatives','bool_masks','derivatives_'\
                                'spm_sub-CSI1_sub-CSI1_mask-RHEarlyVis.nii.gz')
        resample.inputs.master = join(os.path.dirname(__file__),'derivatives','T1',
                                      'sub-CSI1_ses-16_anat_sub-CSI1_ses-16_T1w.nii.gz')
        resample.inputs.out_file = join('temp','test_resample.nii.gz')
        resample.inputs.args = '-DAFNI_NIFTI_TYPE_WARN=NO' 
        resample.run()
        os.remove(resample.inputs.out_file)

    except OSError:
        print('WARNING: AFNI could not be found! Part of the prediction pipeline will not work.\n'\
              'Please install AFNI or use the docker container as described in the README.\n')
        raise 
    try:
        aw = fsl.ApplyWarp()
        aw.inputs.in_file = join(os.path.dirname(__file__),'derivatives','bool_masks','derivatives_'\
                                'spm_sub-CSI1_sub-CSI1_mask-RHEarlyVis.nii.gz')
        aw.inputs.ref_file = join(os.path.dirname(__file__),'derivatives','sub1.anat',
                                  'T1_to_MNI_nonlin.nii.gz')
        aw.inputs.field_file = join(os.path.dirname(__file__),'derivatives','sub1.anat',
                                    'T1_to_MNI_nonlin_coeff.nii.gz')
        aw.inputs.premat = join(os.path.dirname(__file__),'derivatives','sub1.anat',
                                'T1_nonroi2roi.mat')
        aw.inputs.interp = 'nn'
        aw.inputs.out_file = join('temp','test_mni.nii.gz')
        aw.run()
        os.remove(aw.inputs.out_file)
    except OSError:
        print('WARNING: FSL could not be found! Parts of the prediction pipeline will not work.\n'\
              ' Please install FSL or use the docker container as described in the README.\n')
        raise


def _CLI_interface():
    """Command line interface to predict group-level fMRI responses to individual image frames.
    
    Examples
    ----------
    >>> python img2fmri.py [-h] --input input_dir_or_movie [--output output_dir]
                                [--roi_list each roi here] [--sigma sigma_val] [--center_crop 0_or_1]
                                [--predict_movie 0_or_1]
    """
    args = _get_args()
    predict(input_data=args.input[0],
            output_dir=args.output,
            roi_list=args.roi_list,
            predict_movie=args.predict_movie,
            sigma=args.sigma,
            center_crop=args.center_crop)


def _get_args():
    """Defines arguments for command line interface version of prediction."""
    parser = argparse.ArgumentParser(description='Convert images into MNI brains.')
    parser.add_argument('--input', metavar='input', type=str, required=True, nargs=1,
                        help='Path to input (directory or movie file, if predict_movie=1), which '\
                             'contains all images (or the movie) to be processed and predicted')
    parser.add_argument('--output', metavar='output_dir', type=str, default='.',
                        help='output directory where fMRI volumes will be saved.')
    parser.add_argument('--roi_list', metavar='roi_list', type=str, nargs='+',
                        default=['LOC', 'PPA', 'RSC'],
                        help='fMRI ROIs to predict, default: LOC, PPA, RSC')
    parser.add_argument('--sigma', metavar='sigma', type=float, default=1.0,
                        help='sigma for smoothing MNI brains, default=1.0')
    parser.add_argument('--center_crop', metavar='center_crop', type=bool, default=0,
                        help='whether to center crop input images instead of resizing full image '\
                             ' to square. Argument passed as \'--center_crop=1\', default=0')
    parser.add_argument('--predict_movie', metavar='predict_movie', type=bool, default=0,
                        help='whether to predict a fMRI response to movie frames. Argument passed'\
                             ' as \'--predict_move=1\' if true. default=0')
    return parser.parse_args()

if __name__ == '__main__':
    _CLI_interface()
