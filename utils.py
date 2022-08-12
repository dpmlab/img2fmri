"""
Analysis and validation functions for imgtofmri python package tutorial.
Tutorial and analysis found at: https://github.com/dpmlab/imgtofmri
Author: Maxwell Bennett mbb2176@columbia.edu
"""
import os
import glob
from natsort import natsorted
from tqdm import tqdm, trange

import numpy as np
from sklearn.linear_model import LinearRegression
import h5py
import cv2
import nibabel as nib


TQDM_FORMAT = '{l_bar}{bar:10}{r_bar}{bar:-10b}'
MNI_SHAPE = (91, 109, 91)


def get_subj_overlap(rois=['LOC', 'PPA', 'RSC']):
    """
    Returns the indices of an MNI brain volume comprising the logical OR of our subjects' ROIs.
    Works for all 5 ROIs; returns a given voxel index if voxel value (post smoothing) > threshold.
    """
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


def get_subj_overlap_nonsmoothed(rois=['LOC', 'PPA', 'RSC']):
    """
    Returns the indices of an MNI brain volume comprising the logical OR of our subjects' ROIs.
    Works for all 5 ROIs, and returns a given voxel index if voxel value of bool mask > threshold.
    """
    threshold = 0.15
    for roi_idx, roi in enumerate(rois):
        for subj in range(0,3):
            lh = nib.load(f'derivatives/mni_bool_masks/sub{subj+1}_LH{roi}_MNI.nii.gz').get_fdata()
            rh = nib.load(f'derivatives/mni_bool_masks/sub{subj+1}_RH{roi}_MNI.nii.gz').get_fdata()
            LH_mask = lh > np.max(lh) * threshold
            RH_mask = rh > np.max(rh) * threshold

            if (roi_idx == 0) and (subj == 0):
                subject_overlap = LH_mask | RH_mask
            else: 
                subject_overlap = subject_overlap | LH_mask | RH_mask
    return subject_overlap


def extract_frames(input_file, output_dir):
    """ Extracts frames from an input movie at 2Hz and saves them to the output_dir. """
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
            cv2.imwrite(f"{output_dir}/frame{frame_count}.jpg", image) # save frame as JPEG file
            msec_count += msec_per_frame
            frame_count += frames_to_skip
            pbar.update(1)
    vidcap.release()


def load_frames(input_dir):
    """ Loads fMRI responses of frames from input_dir in sorted order and returns a 4D volume """
    sorted_files = natsorted(glob.glob(f'{input_dir}/*'))
    movie_response_shape = np.concatenate((MNI_SHAPE, [len(sorted_files)]))
    movie_response = np.full((movie_response_shape), np.nan)

    desc = "Loading movie frames"
    for i, filename in enumerate(tqdm(sorted_files, bar_format=TQDM_FORMAT, desc=desc)):
        movie_response[...,i] = nib.load(filename).get_fdata()
    return movie_response


##### Pre-processing functions:
def remove_average_activity(a):
    """ removes average activity from analysis """
    b = np.zeros(a.shape)
    for img in trange(a.shape[-1], bar_format=TQDM_FORMAT, desc='Removing average activity'): 
        b[...,img] = a[...,img] - np.mean(a, axis=-1)
    return b


def remove_DCT(a):
    """
    Remove high-pass filtered activity over 120s (period_cut) by regressing out a
    discrete cosine transformation (DCT) basis set.
    """
    period_cut = 120 # threshold for low-pass filter
    timepoints = 168 # TRs
    timestep = 2 
    frametimes = timestep * np.arange(timepoints)
    cdrift = _cosine_drift(period_cut, frametimes)

    regressor = cdrift
    brain_shape = a.shape
    brain = a.reshape((-1, a.shape[-1])) # flatten to (vox, TRs)

    model = LinearRegression()
    model.fit(regressor, brain.T)
    brain = brain - np.dot(model.coef_, regressor.T) - model.intercept_[:, np.newaxis]
    brain = brain.reshape(brain_shape)
    return brain


def conv_hrf_and_downsample(E, TR=2, nTR=168):
    """ Convolve and downsample movie timecourse to TR timecourse """
    T = E.shape[-1]

    # HRF (from AFNI)
    dt = np.arange(0, 15, step=0.5)
    p = 8.6
    q = 0.547
    hrf = np.power(dt / (p * q), p) * np.exp(p - dt / q)

    # Convolve frame timecourse with HRF
    initial_timecourse = np.zeros(E.shape)
    for x in trange(E.shape[0], bar_format=TQDM_FORMAT, desc='Convolving with HRF'):
        initial_timecourse[x,:] = np.convolve(E[x,:], hrf)[:T]

    # Downsample frame timecourse to TR timecourse
    timepoints = np.linspace(0, T, nTR)
    downsampled = np.zeros((E.shape[0], len(timepoints)))
    for x in trange(E.shape[0], bar_format=TQDM_FORMAT, desc='Downsampling'):
        downsampled[x,:] = np.interp(timepoints, np.arange(0, T), initial_timecourse[x,:])

    return downsampled


def regress_out(true_dir):
    """ Remove nuisance artifacts from Partly Cloudy dataset """
    for subj in range(123,156):
        regressor = h5py.File(f'{true_dir}/sub-pixar{subj}/sub-pixar{subj}_task-pixar_run-001_ART'\
                               '_and_CompCor_nuisance_regressors.mat', 'r')
        regressor = regressor['R']
        regressor = np.asarray(regressor).T 

        subj_TRs = f'{true_dir}/sub-pixar{subj}/subj{subj}_r.nii.gz'
        brain_nib = nib.load(subj_TRs)
        brain = brain_nib.get_fdata() # shape: (x,y,z,TRs)

        n_voxels = np.prod(brain.shape[:-1])
        brain = brain.reshape((n_voxels, brain.shape[-1])) # flatten brain: (n_vox, n_TRs)

        model = LinearRegression()
        model.fit(regressor, brain.T) # takes shapes of: (n_samples, n_features; n_TRs, n_vox)
        brain = brain - np.dot(model.coef_, regressor.T) - model.intercept_[:, np.newaxis]
        
        brain = brain.reshape((brain_nib.shape))                    

        nib.save(nib.Nifti1Image(brain, affine=brain_nib.affine),
             f'{true_dir}/sub-pixar{subj}/subj{subj}_r_ro.nii.gz')


def pc_event_bounds(offset = 0):
    """ Used to downsample human annotated event bounds from partly cloudy to TR space """
    # 13 bounds (+2 endpoints) == 14 events
    bounds_seconds = [23, 52, 60+30, 60+53, 120+24, 120+46,
                      180+8, 180+33, 240+3, 240+12, 240+40, 240+53, 300+11]

    bounds_seconds = np.asarray(bounds_seconds)
    bounds_seconds += 6 # add for BOLD signal

    nSec = 5*60 + 45
    nTR = 168
    mov = np.zeros((nSec))
    mov[bounds_seconds] = 100

    timepoints = np.linspace(0, nSec, nTR)
    downsample = np.interp(timepoints, np.arange(0, nSec), mov)
    
    bounds = np.where(downsample > 0)[0]
    bounds -= 2 # offset from beginning credits
    
    bounds += offset
    bounds = np.concatenate(([0], bounds, [nTR]))

    bounds[-1] = nTR - 10 # subtract 10 == -(2 opening credit TRs, 8 ending credit TRs)
    return bounds


# This function has been copied from nipype's confounds.py
def _cosine_drift(period_cut, frametimes):
    """Create a cosine drift matrix with periods greater or equals to period_cut
    Parameters
    ----------
    period_cut: float
         Cut period of the low-pass filter (in sec)
    frametimes: array of shape(nscans)
         The sampling times (in sec)
    Returns
    -------
    cdrift:  array of shape(n_scans, n_drifts)
             cosin drifts plus a constant regressor at cdrift[:,0]
    Ref: http://en.wikipedia.org/wiki/Discrete_cosine_transform DCT-II
    """
    len_tim = len(frametimes)
    n_times = np.arange(len_tim)
    hfcut = 1.0 / period_cut  # input parameter is the period

    # frametimes.max() should be (len_tim-1)*dt
    dt = frametimes[1] - frametimes[0]
    # hfcut = 1/(2*dt) yields len_time
    # If series is too short, return constant regressor
    order = max(int(np.floor(2 * len_tim * hfcut * dt)), 1)
    cdrift = np.zeros((len_tim, order))
    nfct = np.sqrt(2.0 / len_tim)

    for k in range(1, order):
        cdrift[:, k - 1] = nfct * np.cos((np.pi / len_tim) * (n_times + 0.5) * k)

    cdrift[:, order - 1] = 1.0  # or 1./sqrt(len_tim) to normalize
    return cdrift
