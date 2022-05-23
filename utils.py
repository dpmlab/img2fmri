import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression
import pickle
from scipy import stats
import h5py

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image

from pathlib import Path
import argparse
import os
import glob

import nibabel as nib
from nipype.interfaces import afni
from nipype.interfaces import fsl

from natsort import natsorted

from tqdm import tqdm, trange

# smallsize=14; mediumsize=16; largesize=18
# plt.rc('xtick', labelsize=smallsize); plt.rc('ytick', labelsize=smallsize); plt.rc('legend', fontsize=mediumsize)
# plt.rc('figure', titlesize=largesize); plt.rc('axes', labelsize=mediumsize); plt.rc('axes', titlesize=mediumsize)
# ###

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

# Pre-processing functions:
def remove_average_activity(a):
    for img in range(a.shape[-1]): 
        a[...,img] = a[...,img] - np.mean(a, axis=-1)

    return a


def remove_DCT(a):
    period_cut = 120
    timepoints = 168
    timestep = 2
    frametimes = timestep * np.arange(timepoints)
    cdrift = _cosine_drift(period_cut, frametimes)

    # residuals, regressors = cosine_filter(a, 2, 120, remove_mean=True)
    # return residuals

    regressor = cdrift
    brain_shape = a.shape
    brain = a.reshape((-1, a.shape[-1])) # flatten to (vox, TRs)

    model = LinearRegression()
    model.fit(regressor, brain.T)
    brain = brain - np.dot(model.coef_, regressor.T) - model.intercept_[:, np.newaxis]
    brain = brain.reshape(brain_shape) #TODO was 'bran =' ???
#     brain = brain.reshape((a.shape[0], a.shape[1],a.shape[2], brain.shape[-1]))
    # print(f"new brain shape: {brain.shape}")

    return brain


def regress_out(true_dir):
    for subj in range(123,156):
        # if (subj % 10 == 0): print(f"Subj{subj}")
        regressor = h5py.File(f'{true_dir}/sub-pixar{subj}/sub-pixar{subj}_task-pixar_run-001_ART_and_CompCor_nuisance_regressors.mat', 'r')
        regressor = regressor['R']
        regressor = np.asarray(regressor).T # shape: x,168 --> 168,x
        print(f"subj{subj} regressor[0:3]: {regressor[0:3]}")

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


# Convolve and downsample event timecourses to TR timecourses
def regressor_to_TR(E, TR=2, nTR=168):
    T = E.shape[3]
    # print(f"num frames: {T}, num TRs: {nTR}")
    # nEvents = E.shape[1]

    # HRF (from AFNI)
    dt = np.arange(0, 15, step=0.5)
    p = 8.6
    q = 0.547
    hrf = np.power(dt / (p * q), p) * np.exp(p - dt / q)

    # Convolve event matrix to get design matrix
    design_seconds = np.zeros(( E.shape[0], E.shape[1], E.shape[2], E.shape[3] ))

    for x in trange(E.shape[0], desc='Convolving with HRF'):
        for y in range(E.shape[1]):
            for z in range(E.shape[2]):
                design_seconds[x,y,z,:] = np.convolve(E[x, y, z, :], hrf)[:T]

    # Downsample event matrix to TRs
    timepoints = np.linspace(0, T, nTR)
    design = np.zeros((E.shape[0], E.shape[1], E.shape[2], len(timepoints)))
    for x in trange(E.shape[0], desc='Downsampling'):
        for y in range(E.shape[1]):
            for z in range(E.shape[2]):
                design[x,y,z,:] = np.interp(timepoints, np.arange(0, T), design_seconds[x,y,z,:])

    return design


def pc_event_bounds(offset = 0):
    # 13 bounds (+2 endpoints) == 14 events
    bounds_seconds = [23, 52, 60+30, 60+53, 120+24, 120+46,
                      180+8, 180+33, 240+3, 240+12, 240+40, 240+53, 300+11]

    bounds_seconds = np.asarray(bounds_seconds)
    bounds_seconds += 6 # add for BOLD signal

    # print(bounds_seconds)
    nSec = 5*60 + 45
    nTR = 168
    mov = np.zeros((nSec))
    mov[bounds_seconds] = 100

    timepoints = np.linspace(0, nSec, nTR)
    downsample = np.interp(timepoints, np.arange(0, nSec), mov)
    
    bounds = np.where(downsample > 0)[0]
    # print(f"bounds downsampled, +6 for hrf: {bounds}")
    bounds -= 2 # offset from beginning credits
    
    bounds += offset
    bounds = np.concatenate(([0], bounds, [nTR]))

    bounds[-1] = nTR - 10 # subtract 10 == -(2 opening credit TRs, 8 ending credit TRs)
    # print(bounds)

    return bounds


# Returns the indices of an MNI brain volume comprising the logical OR of our subjects' ROIs.
# Works for all 5 ROIs, and returns a given voxel index if voxel value of bool mask > threshold.
def get_subj_overlap_nonsmoothed(rois=['LOC', 'PPA', 'RSC']):
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


def dir_path(string):
    if os.path.isdir(string):
        return Path(string).resolve()
    else:
        raise NotADirectoryError(string)


import cv2
from tqdm import tqdm


def extractImages(pathIn, pathOut):
    vidcap = cv2.VideoCapture(pathIn)
    success, image = vidcap.read()
    if not success:
        print(f"Did not find video at {pathIn}") 
    
    frame_count = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
    framerate = vidcap.get(cv2.CAP_PROP_FPS)
    movie_length = frame_count / framerate
    msec_per_frame = 500 # 2 frames a second
    frames_to_skip = 30
    msec_count = 0
    frame_count = 1
    
    with tqdm(total=int(movie_length * 2), desc='Saving movie frames') as pbar:
        while msec_count < movie_length * 1000:
            vidcap.set(cv2.CAP_PROP_POS_MSEC, msec_count)
            success,image = vidcap.read()
            cv2.imwrite(f"{pathOut}/frame{frame_count}.jpg", image)     # save frame as JPEG file
            msec_count += msec_per_frame
            frame_count += frames_to_skip
            pbar.update(1)

    vidcap.release()

# Plots correlation matrices for Partly Cloudy analyses
def plot_correlation_matrices(real, pred, lum):
    # Correlation matrices
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12,4))
    axes[0].imshow(real)
    axes[1].imshow(pred)
    axes[2].imshow(lum)
    axes[0].set_xlabel('Real brains', fontsize=12)
    axes[1].set_xlabel('Predicted brains', fontsize=12)
    axes[2].set_xlabel('Image luminace', fontsize=12)
    fig.suptitle('Correlation matrices for Partly Cloudy', fontsize=16)

# Calculates the luminance of an image or set of images, and returns a 2-dim object
# with luminance_value x number of frames. Expects an input_dir of image frames
def get_luminance(conv=True, rgb=False, input_dir="partly_cloudy_frames", TRs=168):
    scaler = transforms.Resize((224, 224))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()

    mov = []
    sorted_files = natsorted(glob.glob(f'{input_dir}/*'))
    for i, file in enumerate(tqdm(sorted_files, desc="Loading images")):
        img = Image.open(file)
        t_img = Variable(normalize(to_tensor(scaler(img))))
        image = t_img.data.numpy()
        image = np.transpose(image, (1,2,0))

        mov.append(image)

    mov = np.asarray(mov)
    mov_convd = regressor_to_TR(mov.T, 2, TRs)
    mov_l = mov_convd.T

    y = mov_l[:,:,:,0]*0.2126 + mov_l[:,:,:,1]*0.7152 + mov_l[:,:,:,2]*0.0722
    y = y[2:-8]
    y_flat = y.reshape((y.shape[0], -1))
    y_flat -= np.min(y_flat)
    y_flat /= np.max(y_flat)
    y_flat *= 255

    return y_flat

############################################
# The functions in this section have been copied from nipype's confounds.py
def cosine_filter(
    data, timestep, period_cut, remove_mean=True, axis=-1, failure_mode="error"
):
    datashape = data.shape
    timepoints = datashape[axis]
    if datashape[0] == 0 and failure_mode != "error":
        return data, np.array([])

    data = data.reshape((-1, timepoints))

    frametimes = timestep * np.arange(timepoints)
    X = _full_rank(_cosine_drift(period_cut, frametimes))[0]
    print(f"X.shape: {X.shape}")
    non_constant_regressors = X[:, :-1] if X.shape[1] > 1 else np.array([])

    betas = np.linalg.lstsq(X, data.T)[0]

    if not remove_mean:
        X = X[:, :-1]
        betas = betas[:-1]

    residuals = data - X.dot(betas).T

    return residuals.reshape(datashape), non_constant_regressors


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


def _full_rank(X, cmax=1e15):
    """
    This function possibly adds a scalar matrix to X
    to guarantee that the condition number is smaller than a given threshold.
    Parameters
    ----------
    X: array of shape(nrows, ncols)
    cmax=1.e-15, float tolerance for condition number
    Returns
    -------
    X: array of shape(nrows, ncols) after regularization
    cmax=1.e-15, float tolerance for condition number
    """
    U, s, V = fallback_svd(X, full_matrices=False)
    smax, smin = s.max(), s.min()
    c = smax / smin
    if c < cmax:
        return X, c
    IFLOGGER.warning("Matrix is singular at working precision, regularizing...")
    lda = (smax - cmax * smin) / (cmax - 1)
    s = s + lda
    X = np.dot(U, np.dot(np.diag(s), V))
    return X, cmax

def fallback_svd(a, full_matrices=True, compute_uv=True):
    try:
        return np.linalg.svd(a, full_matrices=full_matrices, compute_uv=compute_uv)
    except np.linalg.LinAlgError:
        pass

    from scipy.linalg import svd

    return svd(a, full_matrices=full_matrices, compute_uv=compute_uv, lapack_driver="gesvd")
############################################