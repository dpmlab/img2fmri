"""
Analysis and validation functions for img2fmri python package tutorial.
Tutorial and analysis found at: https://github.com/dpmlab/img2fmri
Author: Maxwell Bennett mbb2176@columbia.edu
"""
import os
import glob
from natsort import natsorted
from tqdm import tqdm, trange

import numpy as np
from sklearn.linear_model import LinearRegression
import cv2
import nibabel as nib
from nipype.algorithms.confounds import _cosine_drift


TQDM_FORMAT = '{l_bar}{bar:10}| {n}/{total_fmt} [{elapsed}<{remaining}]'
MNI_SHAPE = (91, 109, 91)


def get_subj_overlap(roi_list=['LOC', 'PPA', 'RSC'], threshold=0.15):
    """Returns the indices of an MNI brain volume comprising the logical OR of our subjects' ROIs,
    after smoothing with a sigma=1 (the default smoothing kernel of img2fmri). If using a different
    sigma value with the prediction pipeline, users should retrieve pre-smoothed masks using
    'get_subj_overlap_nonsmoothed' and then smooth those masks with their chosen kernel (see
    img2fmri.smooth_brains() for example smoothing code).

    Parameters
    ----------
    roi_list : list
        List of ROIs to return indices for.
        Defaults to ['LOC', 'PPA', 'RSC'].
    threshold : float
        Lowerbound threshold used to consider a voxel within the mask.
        Defaults to 0.15.

    Returns
    -------
    numpy.ndarray
        3D array of boolean values, with shape of MNI brain, where voxel is true if
    numpy.ndarray
        3D array of boolean values, with shape of MNI brain, where voxel is true if
        value (post smoothing) > threshold.
    """
    if type(roi_list) != list:
        assert False, "Error: argument 'roi_list' must be a list."

    for roi_idx, roi in enumerate(roi_list):
        for subj in range(0,3):
            lh_path = os.path.join(os.path.dirname(__file__),
                                   'derivatives',
                                   's_bool_masks',
                                   f's_sub{subj+1}_LH{roi}_MNI.nii.gz')
            rh_path = os.path.join(os.path.dirname(__file__),
                                   'derivatives',
                                   's_bool_masks',
                                   f's_sub{subj+1}_RH{roi}_MNI.nii.gz')
            lh = nib.load(lh_path).get_fdata()
            rh = nib.load(rh_path).get_fdata()
            LH_mask = lh > np.max(lh) * threshold
            RH_mask = rh > np.max(rh) * threshold

            if (roi_idx == 0) and (subj == 0):
                subject_overlap = LH_mask | RH_mask
            else: 
                subject_overlap = subject_overlap | LH_mask | RH_mask
    return subject_overlap


def get_subj_overlap_nonsmoothed(roi_list=['LOC', 'PPA', 'RSC'], threshold=0.15):
    """Returns the indices of an MNI brain volume comprising the logical OR of our subjects' ROIs,
    before smoothing.

    Parameters
    ----------
    roi_list : list
        List of ROIs to return indices for.
        Defaults to ['LOC', 'PPA', 'RSC'].
    threshold : float
        Lowerbound threshold used to consider a voxel within the mask.
        Defaults to 0.15.

    Returns
    -------
    numpy.ndarray
        3D array of boolean values, with shape of MNI brain, where voxel is true if
        value (pre smoothing) > threshold.
    """
    if type(roi_list) != list:
        assert False, "Error: argument 'roi_list' must be a list."

    threshold = 0.15
    for roi_idx, roi in enumerate(roi_list):
        for subj in range(0,3):
            lh_path = os.path.join(os.path.dirname(__file__),
                                   'derivatives',
                                   'mni_bool_masks',
                                   f'sub{subj+1}_LH{roi}_MNI.nii.gz')
            rh_path = os.path.join(os.path.dirname(__file__),
                                   'derivatives',
                                   'mni_bool_masks',
                                   f'sub{subj+1}_RH{roi}_MNI.nii.gz')
            lh = nib.load(lh_path).get_fdata()
            rh = nib.load(rh_path).get_fdata()
            LH_mask = lh > np.max(lh) * threshold
            RH_mask = rh > np.max(rh) * threshold

            if (roi_idx == 0) and (subj == 0):
                subject_overlap = LH_mask | RH_mask
            else: 
                subject_overlap = subject_overlap | LH_mask | RH_mask
    return subject_overlap


def extract_frames(input_file, output_dir):
    """Extracts frames from an input movie at 2Hz and saves them to the output_dir.

    Parameters
    ----------
    input_file : str
        Relative path to input movie file.
    output_dir : str
        Relative path to output directory to save movie frames.
    """
    vidcap = cv2.VideoCapture(input_file)
    success, image = vidcap.read()
    if not success:
        print(f"Did not find video at {input_file}")
    try:
        os.makedirs(output_dir)
    except FileExistsError:
        pass

    frame_count = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
    framerate = vidcap.get(cv2.CAP_PROP_FPS)
    movie_length = frame_count / framerate
    msec_per_frame = 500 # 2 frames a second
    frames_to_skip = 30
    msec_count = 0
    frame_count = 1

    desc = 'Saving movie frames'
    with tqdm(total=int(movie_length * 2) + 1, bar_format=TQDM_FORMAT, desc=desc) as pbar:
        while msec_count < movie_length * 1000:
            vidcap.set(cv2.CAP_PROP_POS_MSEC, msec_count)
            success,image = vidcap.read()
            if success:
                cv2.imwrite(os.path.join(output_dir,f"frame{frame_count}.jpg"), image) # save as JPG
            msec_count += msec_per_frame
            frame_count += frames_to_skip
            pbar.update(1)
    vidcap.release()


def load_frames(input_dir):
    """Loads predicted fMRI responses to frames in sorted order, and returns a 4D volume.

    Parameters
    ----------
    input_dir : str
        Relative path to directory of frames to load.

    Returns
    -------
    numpy.ndarray
        4D brain volume of shape: (x,y,z,num_frames).
    """
    sorted_files = natsorted(glob.glob(os.path.join(input_dir,'*')))
    movie_response_shape = np.concatenate((MNI_SHAPE, [len(sorted_files)]))
    movie_response = np.zeros((movie_response_shape))

    desc = "Loading movie frames"
    for i, filename in enumerate(tqdm(sorted_files, bar_format=TQDM_FORMAT, desc=desc)):
        movie_response[...,i] = nib.load(filename).get_fdata()
    return movie_response


##### Pre-processing functions:
def remove_average_activity(input_brain):
    """Removes average (across time) activity from 4D brain volume.

    Parameters
    ----------
    input_brain : numpy.ndarray
        4D brain volume of shape: (x,y,z,num_frames).

    Returns
    -------
    numpy.ndarray
        4D brain volume of shape: (x,y,z,num_frames).
    """
    out = np.zeros(input_brain.shape)
    desc = 'Removing average activity'
    for img in trange(input_brain.shape[-1], bar_format=TQDM_FORMAT, desc=desc): 
        out[...,img] = input_brain[...,img] - np.mean(input_brain, axis=-1)
    return out


def remove_DCT(input_brain, period_cut=120):
    """High-pass filter brain activity (default is a period_cut of constant activity over 120s)
    by regressing out a discrete cosine transformation (DCT) basis set.

    Parameters
    ----------
    input_brain : numpy.ndarray
        4D brain volume of shape: (x,y,z,num_frames).
    period_cut : int
        threshold (seconds) of high-pass filter. Defaults to 120.

    Returns
    -------
    numpy.ndarray
        4D brain volume of shape: (x,y,z,num_frames).
    """
    timestep = 2 # timepoints per second
    num_TR = input_brain.shape[-1] # number of TRs
    frametimes = timestep * np.arange(num_TR)
    cdrift = _cosine_drift(period_cut, frametimes)

    regressor = cdrift
    brain_shape = input_brain.shape
    brain = input_brain.reshape((-1, input_brain.shape[-1])) # flatten to (vox, TRs)

    model = LinearRegression()
    model.fit(regressor, brain.T)
    brain = brain - np.dot(model.coef_, regressor.T) - model.intercept_[:, np.newaxis]
    brain = brain.reshape(brain_shape)
    return brain


def conv_hrf_and_downsample(input_brain, num_TR, TR=2):
    """Convolve with hemodynamic response function (HRF) and downsample from movie frame timecourse
    to TR timecourse.

    Parameters
    ----------
    input_brain : numpy.ndarray
        4D brain volume of shape: (x,y,z,num_frames).
    num_TR : int
        Number of TRs to downsample to.
    TR : int
        Temporal resolution, or, frames per second (Hz). Defaults to 2.

    Returns
    -------
    numpy.ndarray
        4D brain volume of shape: (x,y,z,num_TRs).
    """
    num_frames = input_brain.shape[-1]

    # HRF (from AFNI)
    dt = np.arange(0, 15, step=0.5)
    p = 8.6
    q = 0.547
    hrf = np.power(dt / (p * q), p) * np.exp(p - dt / q)

    # Convolve frame timecourse with HRF
    initial_timecourse = np.zeros(input_brain.shape)
    for x in trange(input_brain.shape[0], bar_format=TQDM_FORMAT, desc='Convolving with HRF'):
        initial_timecourse[x,:] = np.convolve(input_brain[x,:], hrf)[:num_frames]

    # Downsample frame timecourse to TR timecourse
    timepoints = np.linspace(0, num_frames, num_TR)
    downsampled = np.zeros((input_brain.shape[0], len(timepoints)))
    for x in trange(input_brain.shape[0], bar_format=TQDM_FORMAT, desc='Downsampling'):
        downsampled[x,:] = np.interp(timepoints, np.arange(0, num_frames), initial_timecourse[x,:])

    return downsampled