import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.legend_handler import HandlerTuple
import pandas as pd
from sklearn.linear_model import Ridge
from joblib import dump, load
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
import pickle

import nibabel as nib
from nipype.interfaces import afni
from nipype.interfaces import fsl

from natsort import natsorted

import zipfile
import wget

from utils import get_subj_overlap, regressor_to_TR

from tqdm import tqdm, trange

ALL_ROIS = ["EarlyVis", "OPA", "LOC", "RSC", "PPA"]
TWINSET_CATEGORIES = ['animals', 'objects', 'scenes', 'people', 'faces']
PC_EVENT_BOUNDS = [  0,  12,  26,  44,  56,  71,  81,  92, 104, 119, 123, 136, 143, 151]

def generate_activations(input_dir):
    output = f"temp/activations/"

    # Default input image transformations for ImageNet
    scaler = transforms.Resize((224, 224))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()

    # Load the pretrained model, set to eval mode
    model = models.resnet18(pretrained=True)
    model.eval()

    for filename in tqdm(glob.glob(f"{input_dir}/*"), desc='Pushing images through CNN'):
        # TODO need to have a check for non jpg/pngs!
        if Path(filename).suffix != ".jpg" and Path(filename).suffix != ".png":
            # print(f"skipping {filename} with suffix: {Path(filename).suffix}")
            continue

        img = Image.open(filename)
        t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))

        # Create network up to last layer, push image through, flatten
        layer_extractor = torch.nn.Sequential(*list(model.children())[:-1])
        feature_vec = layer_extractor(t_img).data.numpy().squeeze()
        feature_vec = feature_vec.flatten()
        
        # Save image
        image_name = Path(filename).stem
        np.save(f"{output}{image_name}.npy", feature_vec)
        img.close()

    num_images = len(glob.glob(f"{input_dir}/*.png") + glob.glob(f"{input_dir}/*.jpg"))
    # tqdm.write(f"Saved: CNN activations of {num_images} images")


def generate_brains(roi_list=["LOC", "RSC", "PPA"]):
    # roi_list = ["EarlyVis", "OPA", "LOC", "RSC", "PPA"]
    # roi_list = ["LOC", "RSC", "PPA"]
    ridge_p_grid = {'alpha': np.logspace(1, 5, 10)}

    model_dir = f"models/"
    shape_array = np.load('derivatives/shape_array.npy', allow_pickle=True)

    # For each subject, for each input file, predict all ROIs for that subject and save prediction
    total_num_images = 3 * len(glob.glob('temp/activations/*'))
    with tqdm(total=total_num_images, desc='Predicting images into subjects\' brains') as pbar:
        for subj in range(0,3):
            for filename in glob.glob('temp/activations/*'):
                pbar.update(1)
                actv  = np.load(open(filename, 'rb'), allow_pickle=True)

                for roi_idx, roi in enumerate(roi_list):
                    # print(roi_idx, roi, f"index in list: {ALL_ROIS.index(roi)}")

                    model = load(f'models/subj{subj+1}_{roi}_model.pkl')
                    y_pred_brain = model.predict([actv])
                    brain = y_pred_brain[0]

                    # Get left hemisphere
                    T1_mask_nib = nib.load(f"derivatives/bool_masks/derivatives_spm_sub-CSI{subj+1}" \
                                        f"_sub-CSI{subj+1}_mask-LH{roi}.nii.gz")
                    T1_mask_shape = T1_mask_nib.header.get_data_shape()[0:3]
                    LH_T1_mask = T1_mask_nib.get_fdata() > 0  # 3D boolean array

                    # Get right hemisphere
                    T1_mask_nib = nib.load(f"derivatives/bool_masks/derivatives_spm_sub-CSI{subj+1}" \
                                        f"_sub-CSI{subj+1}_mask-RH{roi}.nii.gz")                            
                    RH_T1_mask = T1_mask_nib.get_fdata() > 0  # 3D boolean array

                    # Initialize subj's 3d volume if first time through
                    if (roi_idx == 0):
                        subj_brain = np.empty(T1_mask_shape)
                        subj_brain[:, :, :] = np.NaN  

                    # LH Nanmean
                    a = np.array([subj_brain[LH_T1_mask], 
                                brain[:int(shape_array[subj][ALL_ROIS.index(roi)][0])]]) # nanmean of new w existing
                    a = np.nanmean(a, axis=0)
                    subj_brain[LH_T1_mask] = a

                    # RH Nanmean
                    a = np.array([subj_brain[RH_T1_mask],
                                brain[int(shape_array[subj][ALL_ROIS.index(roi)][0]):]])
                    a = np.nanmean(a, axis=0)
                    subj_brain[RH_T1_mask] = a

                nib.save(nib.Nifti1Image(subj_brain, affine=T1_mask_nib.affine),
                        f'temp/subj_space/sub{subj+1}_{Path(filename).stem}.nii.gz')
    # tqdm.write(f"Saved: Predictions into subjects' brains")
    # Probably should save these for the future in a specific folder to this run, so don't do 
    # everything everytime


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
        # tqdm.write("Saved: Brains in MNI space")


def smooth_brains(sig=1):
    filename = f"temp/mni/*"
    for file in tqdm(glob.glob(filename), desc='Smoothing brains:'):

        stem = Path(file).stem

        out = f"temp/mni_s/{stem}.gz"
        smooth = fsl.IsotropicSmooth()
        smooth.inputs.in_file = file
        smooth.inputs.sigma = sig
        smooth.inputs.out_file = out
        smooth.run()
    # tqdm.write("Saved: Smoothed MNI brains")


# nanmean version:
# def average_subjects(input_dir, output_dir):
#     for filename in glob.glob(f'{input_dir}/*'):
#         if Path(filename).suffix != ".jpg" and Path(filename).suffix != ".png":
#             # print(f"skipping {filename} with suffix: {Path(filename).suffix}")
#             continue
            
#         stem = Path(filename).stem
#         subj_mask_nib = nib.load(f'temp/mni_s/sub1_{stem}.nii.gz')
#         subj_brain = np.full((subj_mask_nib.shape), np.nan)

#         for subj in range(1,4):
#             filename = f'temp/mni_s/sub{subj}_{stem}.nii.gz'
#             brain = nib.load(filename).get_fdata()
#             subj_brain = np.nansum([subj_brain, brain], axis=0)
        
#         subj_brain /= 3 # num_subjects
#         nib.save(nib.Nifti1Image(subj_brain, affine=subj_mask_nib.affine),
#                                  f'{output_dir}/{stem}.nii.gz')
#     print("Saved: Averaged MNI brains")

# zscore version:
def average_subjects(input_dir, output_dir):
    overlap = get_subj_overlap()

    for filename in tqdm(glob.glob(f'{input_dir}/*'), desc='Averaging MNI brains'):
        if Path(filename).suffix != ".jpg" and Path(filename).suffix != ".png":
            # print(f"skipping {filename} with suffix: {Path(filename).suffix}")
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
    # tqdm.write("Saved: Averaged MNI brains")


def predict(input_dir, output_dir, roi_list=['LOC', 'PPA', 'RSC']):
    make_directories()
    generate_activations(input_dir)
    generate_brains(roi_list) # need to be providing the roi_list here
    transform_to_MNI()
    smooth_brains()
    average_subjects(input_dir, output_dir)


def load_imgs(url):
    filename = wget.download(url)

    img_folder = 'input_images' # TODO ensure dir has been created
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall('input_images')
    print(f"Downloaded and extracted: {filename}")


def make_directories():
    temp_dirs = ['temp/activations', 'temp/subj_space', 
                 'temp/temp', 'temp/mni', 'temp/mni_s']
    for directory in temp_dirs:
        try:
            os.makedirs(directory)
        except FileExistsError as e:
            for f in glob.glob(f"{directory}/*"):
                os.remove(f)


# Twinset:
#   Group correlations:
def twinset_generate_group_correlations(output_dir):
    corr = np.zeros((156, 156))
    overlap = get_subj_overlap()

    for i_pred, image_pred in enumerate(tqdm(range(1,157), desc='Generating correlations with group-averaged brains')):
        # if i_pred % 25 == 0 and i_pred > 0: print(f"completed {i_pred} images")
        pred_brain = nib.load(f'{output_dir}/{"{:03d}".format(image_pred)}.nii.gz').get_fdata()
        
        for i_test, image_test in enumerate(range(1,157)):
            true_brain = nib.load(f'../twinset/avgs/avg_beta_{"{:04d}".format(image_test)}.nii.gz').get_fdata()
            pred_overlap = pred_brain[overlap] 
            true_overlap = true_brain[overlap]

            pearson_r = stats.pearsonr(pred_overlap, true_overlap)[0] #just get r, not p val
            corr[i_pred, i_test] = pearson_r


    pkl_filename = f"../twinset/corr_avg_brains.pkl"
    pickle.dump(corr, open(pkl_filename, 'wb'))

def twinset_random_group_permutations(n_shuffle=1000, print_vals=False):
    corr = np.zeros((156, 156))
    filename = f"../twinset/corr_avg_brains.pkl"
    corr = np.load(filename, allow_pickle = True)

    df = corr
    num_resample = n_shuffle
    offdiag = (1-np.diag(np.ones(156))).astype(bool)
    real_diff = np.diag(df).mean() - df[offdiag].mean()

    resampled = np.zeros((num_resample, 2))
    diffs = np.zeros(num_resample)
    count_over_real = 0
    for i in range(num_resample):
        df = pd.DataFrame(corr)
        df = df.sample(frac=1).reset_index(drop=True)
        df = pd.DataFrame.to_numpy(df)

        resampled[i, 0] = np.diag(df).mean()
        resampled[i, 1] = df[offdiag].mean()
        diffs[i] = np.diag(df).mean() - df[offdiag].mean()
        if np.diag(df).mean() - df[offdiag].mean() >= real_diff:
            count_over_real += 1

    p = (count_over_real + 1)/(num_resample + 1)
    n = (stats.norm.sf((real_diff - np.mean(diffs))/ (np.std(diffs))))
    
    if print_vals:
        print(f"Real difference:      {real_diff:.6f}")
        print(f"Mean null difference: {np.mean(diffs):.6f}")
        print(f"p value:              {p:.6f}")
        print(f"norm.sf:              {n:.6f}\n")
    #  p = ((# perms >= real) + 1)/(# perms + 1)
    #  p may be closer to norm.sf((Real - null mean)/(null std))

    # Plotting
    fig, ax = plt.subplots(figsize=(7,7))
    ax.scatter(1, real_diff, marker='o', color='k', edgecolors='k', label='Real difference')
    plotparts = ax.violinplot(diffs, showextrema=False,)
    plotparts['bodies'][0].set_edgecolor('cornflowerblue')
    plotparts['bodies'][0].set_facecolor('cornflowerblue')
    plotparts['bodies'][0].set_alpha(.7)
    plotparts['bodies'][0].set_label("Null distribution")
    ax.set_xticks([1])
    ax.set_xlim(0.5,1.5)
    ax.set_xticklabels(["Group average"], fontsize=12)
    ax.set_ylabel("Diagonal - off-diagonal", fontsize=12)
    ax.set_title("Difference between diagonal and off-diagonal \n for correlation matrix of real and predicted brains")
    max_y = max(diffs.max(), real_diff)
    # y_height = max(diffs.max(), real_diff) * 1.05
    # y_lower, y_upper = ax.get_ylim()
    # ax.set_ylim(y_lower, y_upper * 1.08)

    plot_significance_asterisk(ax.get_xticks()[0], max_y,  p, fs=16)
    # barplot_annotate_brackets(0, 1, 'p < 0.0001', [.75, 1.25], [0.048, 0.048], fs=12)
    ax.legend(loc="lower right", fontsize=12)




#   Category correlations:
def twinset_generate_category_correlations(output_dir):
    corr = np.zeros((5,), dtype=object)
    overlap = get_subj_overlap()

    for cat_idx, cat_range in enumerate([range(1,29), range(29,65), range(65,101), range(101, 125), range(125,157)]):
        cat_length = cat_range[-1] - cat_range[0] + 1
        cat_corr = np.zeros((cat_length, cat_length))
        for i_pred, image_pred in enumerate(tqdm(cat_range, desc=f'Generating correlations for \'{TWINSET_CATEGORIES[cat_idx]}\'')):
            pred_brain = nib.load(f'{output_dir}/{"{:03d}".format(image_pred)}.nii.gz').get_fdata()
            # if i_pred % 20 == 0: print(f"completed {i_pred} images")
            
            for i_test, image_test in enumerate(cat_range):
                true_brain = nib.load(f'../twinset/avgs/avg_beta_{"{:04d}".format(image_test)}.nii.gz').get_fdata()
                pred_overlap = pred_brain[overlap] 
                true_overlap = true_brain[overlap]

                pearson_r = stats.pearsonr(pred_overlap, true_overlap)[0] #just get r, not p val
                cat_corr[i_pred, i_test] = pearson_r
                
        corr[cat_idx] = cat_corr
        # print(f"Completed category: {TWINSET_CATEGORIES[cat_idx]}")

    pkl_filename = f"../twinset/corr_categories.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(corr, file)

def twinset_random_category_permutations(n_shuffle=1000, print_vals=False):
    filename = f"../twinset/corr_categories.pkl"
    full_corr = np.load(filename, allow_pickle=True)
    num_cat = full_corr.shape[0]
    ranked_full = np.empty((num_cat,),dtype=object)
    num_resample = n_shuffle
    shuffled_diffs = np.empty((num_cat,num_resample,), dtype=float)
    real_diffs = np.empty((num_cat,), dtype=object)
    pstats = np.empty((num_cat,2))

    for c in range(num_cat):
        df = full_corr[c] # get specific corr matrix for category
        offdiag = (1-np.diag(np.ones(df.shape[0]))).astype(bool)
        real_difference = np.diag(df).mean() - df[offdiag].mean()
        real_diffs[c] = real_difference
        resampled = np.zeros((num_resample, 2))
        diffs = np.zeros(num_resample)
        count_over_real = 0
        for i in range(num_resample):
            df = pd.DataFrame(full_corr[c]) # reset dataframe
            df = df.sample(frac=1).reset_index(drop=True) # take random sample
            df = pd.DataFrame.to_numpy(df)

            resampled[i, 0] = np.diag(df).mean()
            resampled[i, 1] = df[offdiag].mean()
            diffs[i] = np.diag(df).mean() - df[offdiag].mean()
            if np.diag(df).mean() - df[offdiag].mean() >= real_difference:
                count_over_real += 1
        
        p = (count_over_real + 1)/(num_resample + 1)
        n = (stats.norm.sf((real_difference - np.mean(diffs))/ (np.std(diffs))))
        shuffled_diffs[c] = diffs
        pstats[c,0] = p
        pstats[c,1] = n

    if print_vals:
        for c in range(num_cat):
            print(f"Real difference:      {real_diffs[c]:.6f}")
            print(f"Mean null difference: {np.mean(diffs[c]):.6f}")
            print(f"p value:              {pstats[c,0]:.6f}\n")

    categories = ['animals', 'objects', 'scenes', 'people', 'faces']
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    fig, ax = plt.subplots(figsize=(7,7))
    plotparts = ax.violinplot(shuffled_diffs.T, showextrema = False)
    for i, p in enumerate(plotparts['bodies']):
        p.set_edgecolor(colors[i])
        p.set_facecolor(colors[i])
        p.set_alpha(.5)
    
    ax.scatter(range(1,6), real_diffs, marker='o', color='k', edgecolors='k', label='Real matrix')
    ax.set_title("Difference between diagonal and off-diagonal \n for correlation matrix of real and predicted brains")
    ax.set_ylabel("Diagonal - off-diagonal", fontsize=12)
    ax.set_xticks(range(1,6))
    ax.set_xticklabels(categories, fontsize=12)
    for i, xtick, p in zip(range(num_cat), ax.get_xticks(), pstats[:,0]):
        max_y = max(real_diffs[i].max(), shuffled_diffs[i].max())
        plot_significance_asterisk(xtick, max_y, p, fs=16)

# ------ append the multicolor category patches
    h, l = ax.get_legend_handles_labels()
    h.append(MulticolorPatch(colors, alpha=0.5))
    l.append("Null distribution")
    ax.legend(h, l, loc="lower left", fontsize=12,
             handler_map={MulticolorPatch: MulticolorPatchHandler()})




def twinset_generate_participant_correlations(output_dir):
    # correlations between predicted brains and true brains (both doubly zscored as averaging)
    # TODO proper file/folder routing
    corr = np.zeros((15,), dtype=object)
    overlap = get_subj_overlap()
    num_images = 156

    total_num_images = 15 * num_images
    with tqdm(total=total_num_images, desc='Generating correlations per participant') as pbar:
        for subj in range(0,15):
            subj_corr = np.zeros((num_images, num_images))
            
            for i_pred, image_pred in enumerate(range(1,num_images+1)): #  TODO no need for enumerates since ranges
                pbar.update(1)
                # if i_pred % 20 == 0: print(f"completed {i_pred} images")
                    
                pred_brain = nib.load(f'{output_dir}/{"{:03d}".format(image_pred)}.nii.gz').get_fdata()
                for i_test, image_test in enumerate(range(1,num_images+1)):
                    filename = f'../twinset/subj{"{:02d}".format(subj+1)}/r_m_s_beta_{"{:04d}".format(image_test)}.nii.gz'
                    true_brain = nib.load(filename).get_fdata()
                    pred_overlap = pred_brain[overlap] 
                    true_overlap = true_brain[overlap]

                    pearson_r = stats.pearsonr(pred_overlap, true_overlap)[0] #just get r, not p val
                    subj_corr[i_pred, i_test] = pearson_r
                    
            corr[subj] = subj_corr

    pkl_filename = f"../twinset/per_subj_correlations.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(corr, file)

def twinset_random_participant_permutations(n_shuffle=1000, print_vals=False):
    filename = f"../twinset/per_subj_correlations.pkl"
    full_corr = np.load(filename, allow_pickle=True)
    num_cat = full_corr.shape[0]
    ranked_full = np.empty((num_cat,),dtype=object)

    num_resample = n_shuffle
    shuffled_diffs = np.empty((num_cat,num_resample,), dtype=float)
    real_diffs = np.empty((num_cat,), dtype=object)
    pstats = np.empty((num_cat,2))

    for c in range(num_cat):

        df = full_corr[c] # get specific corr matrix for category
        offdiag = (1-np.diag(np.ones(df.shape[0]))).astype(bool)
        real_difference = np.diag(df).mean() - df[offdiag].mean()
        real_diffs[c] = real_difference
        resampled = np.zeros((num_resample, 2))
        diffs = np.zeros(num_resample)
        count_over_real = 0
        for i in range(num_resample):
            df = pd.DataFrame(full_corr[c]) # reset dataframe
            df = df.sample(frac=1).reset_index(drop=True) # take random sample
            df = pd.DataFrame.to_numpy(df)

            resampled[i, 0] = np.diag(df).mean()
            resampled[i, 1] = df[offdiag].mean()
            diffs[i] = np.diag(df).mean() - df[offdiag].mean()
            if np.diag(df).mean() - df[offdiag].mean() >= real_difference:
                count_over_real += 1

        p = (count_over_real + 1)/(num_resample + 1)
        n = (stats.norm.sf((real_difference - np.mean(diffs))/ (np.std(diffs))))
        shuffled_diffs[c] = diffs
        pstats[c,0] = p
        pstats[c,1] = n

    if print_vals:
        for c in range(num_cat):
            print(f"Real difference:      {real_diffs[c]:.6f}")
            print(f"Mean null difference: {np.mean(diffs[c]):.6f}")
            print(f"p value:              {pstats[c,0]:.6f}\n")

    fig, ax = plt.subplots(figsize=(7,7))
    ax.scatter(range(1,16), real_diffs, marker='o', color='k', edgecolors='k', label='Real matrix')
    plotparts = ax.violinplot(shuffled_diffs.T, showextrema = False)
    for i, p in enumerate(plotparts['bodies']):
        p.set_facecolor("cornflowerblue")
        p.set_edgecolor("cornflowerblue")
        p.set_alpha(.5)
    p.set_label("Null distribution")
    ax.set_xticks(range(1,16))
    ax.set_xticklabels([f"{i}" for i in range(1,16)], fontsize=12)

    ax.set_ylabel("Diagonal - off-diagonal", fontsize=12)
    ax.set_xlabel("Participant", fontsize=12)
    ax.legend(loc="lower right", fontsize=12)
    ax.set_title("Difference between diagonal and off-diagonal \n for correlation matrix of real and predicted brains")
    
    same_height = True
    for i, xtick, p in zip(range(num_cat), ax.get_xticks(), pstats[:,0]):
        if same_height: 
            max_y = max(real_diffs[:].max(), shuffled_diffs[:].max())
        else:
            max_y = max(real_diffs[i].max(), shuffled_diffs[i].max())
        
        plot_significance_asterisk(xtick, max_y, p, fs=12)


def plot_significance_asterisk(x, max_y, data, maxasterix=None, fs=12):
    
    if type(data) is str:
        text = data
    else:
        # * is p < 0.05
        # ** is p < 0.01
        # *** is p < 0.001
        # etc.
        text = ''
        p = .05
        if data < p:
            text += "*"

        p = 0.01
        while data < p:
            text += '*'
            p /= 10.

            if maxasterix and len(text) == maxasterix:
                break

#         if len(text) == 0:
#             text = 'n. s.'
            
    y = max_y * 1.05
    ax = plt.gca()
    y_lower, y_upper = ax.get_ylim()
    if y * 1.08 > y_upper:
        ax.set_ylim(y_lower, y_upper * 1.08)
    ax.text(x, y, text, ha='center', fontsize=fs)

# pulled from https://stackoverflow.com/questions/11517986/indicating-the-statistically-significant-difference-in-bar-graph
def barplot_annotate_brackets(num1, num2, data, center, height, yerr=None, dh=.05, barh=.05, fs=None, maxasterix=None):
    """ 
    Annotate barplot with p-values.

    :param num1: number of left bar to put bracket over
    :param num2: number of right bar to put bracket over
    :param data: string to write or number for generating asterixes
    :param center: centers of all bars (like plt.bar() input)
    :param height: heights of all bars (like plt.bar() input)
    :param yerr: yerrs of all bars (like plt.bar() input)
    :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
    :param barh: bar height in axes coordinates (0 to 1)
    :param fs: font size
    :param maxasterix: maximum number of asterixes to write (for very small p-values)
    """

    if type(data) is str:
        text = data
    else:
        # * is p < 0.05
        # ** is p < 0.005
        # *** is p < 0.0005
        # etc.
        text = ''
        p = .05

        while data < p:
            text += '*'
            p /= 10.

            if maxasterix and len(text) == maxasterix:
                break

        if len(text) == 0:
            text = 'n. s.'

    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]

    if yerr:
        ly += yerr[num1]
        ry += yerr[num2]

    ax_y0, ax_y1 = plt.gca().get_ylim()
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)

    y = max(ly, ry) + dh

    barx = [lx, lx, rx, rx]
    bary = [y, y+barh, y+barh, y]
    mid = ((lx+rx)/2, y+barh+.0001)
#     mid = ((lx+rx)/2, y+barh)

    plt.plot(barx, bary, c='black')

    kwargs = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs['fontsize'] = fs

    plt.text(*mid, text, **kwargs)

# define an object that will be used by the legend
class MulticolorPatch(object):
    def __init__(self, colors, alpha=1):
        self.colors = colors
        self.alpha = alpha
        
# define a handler for the MulticolorPatch object
class MulticolorPatchHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        width, height = handlebox.width, handlebox.height
        patches = []
        for i, c in enumerate(orig_handle.colors):
            patches.append(plt.Rectangle([width/len(orig_handle.colors) * i - handlebox.xdescent, 
                                          -handlebox.ydescent],
                           width / len(orig_handle.colors),
                           height, 
                           facecolor=c, 
                           edgecolor='none',
                           alpha=orig_handle.alpha))

        patch = PatchCollection(patches,match_original=True)

        handlebox.add_artist(patch)
        return patch


def get_luminance(conv=True, rgb=False, input_dir="../partly_cloudy/"):
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
    mov_convd = regressor_to_TR(mov.T, 2, 168)
    mov_l = mov_convd.T

    y = mov_l[:,:,:,0]*0.2126 + mov_l[:,:,:,1]*0.7152 + mov_l[:,:,:,2]*0.0722
    y = y[2:-8]
    y_flat = y.reshape((y.shape[0], -1))
    y_flat -= np.min(y_flat)
    y_flat /= np.max(y_flat)
    y_flat *= 255

    return y_flat


def pc_bootstrapped_pred_lum(bound_averages):
    # plt.figure(figsize=(5, 5))
    nTR = bound_averages.shape[1]
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
    plt.tick_params(axis='x', top=False, labeltop=False)

    sorted_arr = np.sort(bound_averages, axis=2)
    axes[0].plot(bound_averages[0], color='blue', alpha = 0.008,)
    axes[0].plot(bound_averages[1], color='green', alpha = 0.008,)
    axes[1].fill_between(range(21), sorted_arr[0,:,50], sorted_arr[0,:,-50], alpha=0.5, color="blue")
    axes[1].fill_between(range(21), sorted_arr[1,:,50], sorted_arr[1,:,-50], alpha=0.5, color="green")


    ticks = [i for i in range(0, 21)][::2]
    ticklabels = [f'{i:.0f}' for i in range(-10, 11)][::2]
    for ax in axes.ravel():
        ax.set_ylim([-1,1])
        ax.set_xlim([0,20])
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticklabels)
        ax.axhline(0, 0, nTR, linestyle='dashed', color='grey')
        ax.vlines(10, -1, 1, linestyle='dashed', color='grey', alpha=1)

    # plt.legend(fontsize='large', loc='lower right')
    # fig.suptitle("""Boundary triggered average for bootstrapped brains and boundaries
    # btw corr of TRxTR matrices, all TRs predicted(blue) luminance(green)
    # skipping ±2TRs from bounds\n""", y=1.03, fontsize=14)
    plt.show()

def pc_bootstrapped_pred_lum_3TRs(bound_averages):
    nTR = bound_averages.shape[1]
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
    plt.tick_params(axis='x', top=False, labeltop=False)

    sorted_arr = np.sort(bound_averages, axis=2)
    axes[0].plot(bound_averages[0,7:14], color='blue', alpha = 0.008,
                label=f'avg corr(pred,true), mean: {np.mean(bound_averages[:,0,:]):.3f}')
    axes[0].plot(bound_averages[1,7:14], color='green', alpha = 0.008,
                label=f'avg corr(lum,true), mean: {np.mean(bound_averages[:,1,:]):.3f}')
    axes[1].fill_between(range(7), sorted_arr[0,7:14,50], sorted_arr[0,7:14,-50], alpha=0.5, color="blue")
    axes[1].fill_between(range(7), sorted_arr[1,7:14,50], sorted_arr[1,7:14,-50], alpha=0.5, color="green")


    ticks = [i for i in range(0, 7)]
    ticklabels = [f'{i:.0f}' for i in range(-3, 4)]
    for ax in axes.ravel():
        ax.set_ylim([-1,1])
        ax.set_xlim([0,6])
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticklabels)
        ax.axhline(0, 0, nTR, linestyle='dashed', color='grey')
        ax.vlines(3, -1, 1, linestyle='dashed', color='grey', alpha=1)
    # plt.legend(fontsize='large', loc='lower right')
    # fig.suptitle("""Boundary triggered average for bootstrapped brains and boundaries
    # btw corr of TRxTR matrices, all TRs predicted(blue) luminance(green)
    # skipping ±2TRs from bounds\n""", y=1.03, fontsize=14)
    plt.show()

def pc_bootstrapped_difference(bound_averages):
    # plt.figure(figsize=(5, 5))
    nTR = bound_averages.shape[1]
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
    plt.tick_params(axis='x', top=False, labeltop=False)

    sorted_arr = np.sort(bound_averages, axis=2)
    axes[0].plot(bound_averages[0]-bound_averages[1], color='red', alpha = 0.008,)
    axes[1].fill_between(range(21),
                        sorted_arr[0,:,50]-sorted_arr[1,:,50],
                        sorted_arr[0,:,-50]-sorted_arr[1,:,-50], alpha=0.5, color="red")

    ticks = [i for i in range(0, 21)][::2]
    ticklabels = [f'{i:.0f}' for i in range(-10, 11)][::2]
    for ax in axes.ravel():
        ax.set_ylim([-1,1])
        ax.set_xlim([0,20])
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticklabels)
        ax.axhline(0, 0, nTR, linestyle='dashed', color='grey')
        ax.vlines(10, -1, 1, linestyle='dashed', color='grey', alpha=1)

    # plt.legend(fontsize='large', loc='lower right')
    # fig.suptitle("""Boundary triggered average for bootstrapped brains and boundaries
    # btw corr of TRxTR matrices, all TRs predicted-luminance
    # skipping ±2TRs from bounds\n""", y=1.03, fontsize=14)
    plt.show()

def pc_bootstrapped_difference_3TRs(bound_averages):
    # plt.figure(figsize=(5, 5))
    nTR = bound_averages.shape[1]
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
    plt.tick_params(axis='x', top=False, labeltop=False)

    sorted_arr = np.sort(bound_averages, axis=2)
    axes[0].plot(bound_averages[0,7:14]-bound_averages[1,7:14], color='red', alpha = 0.008,)
    axes[1].fill_between(range(7),
                        sorted_arr[0,7:14,50]-sorted_arr[1,7:14,50],
                        sorted_arr[0,7:14,-50]-sorted_arr[1,7:14,-50], alpha=0.5, color="red")

    ticks = [i for i in range(0, 7)]
    ticklabels = [f'{i:.0f}' for i in range(-3, 4)]
    for ax in axes.ravel():
        ax.set_ylim([-1,1])
        ax.set_xlim([0,6])
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticklabels)
        ax.axhline(0, 0, nTR, linestyle='dashed', color='grey')
        # plt.vlines(PC_EVENT_BOUNDS, -.5,.5, linestyle='dashed', color='red', alpha=0.5, label="event boundaries (hum annotated)")
        ax.vlines(3, -1, 1, linestyle='dashed', color='grey', alpha=1)

    # fig.suptitle("""Boundary triggered average for bootstrapped brains and boundaries
    # btw corr of TRxTR matrices, all TRs predicted-luminance
    # skipping ±2TRs from bounds\n""", y=1.03, fontsize=14)
    plt.show()

def generate_bootstrapped_correlations(true, pred, lum, TR_band=None,
                                       true_dir="../pc_true/preprocessed_data"):
    if TR_band != None:
        band = get_TR_band(TR_band, nTR=true.shape[0])

    # LOAD Bootstraps instead of generating
    # set up corr structure for shape: (num_bootstraps, 2 (lum, boot_real; pred, boot_real) x TR)
    init_subj = 123 # For initializing brain
    init_subj_TRs = f'{true_dir}/sub-pixar{init_subj}/r_bold.nii.gz'
    init_subj_nib = nib.load(init_subj_TRs)
    overlap = get_subj_overlap()

    num_bootstraps = 100
    nTR = lum.shape[0]
    notdiag = 1-np.diag(np.ones(158)).astype(bool)
    notdiag = notdiag.astype(bool)
    corr = np.zeros((num_bootstraps, 2, nTR))

    # make bootstrap TODO: tqdm
    for i, b in enumerate(tqdm(range(num_bootstraps), desc='Generating and loading bootstraps')):
    #------- save bootstrap brains if don't exist
        if glob.glob(f'{true_dir}/bootstraps/bootstrap_{b}.nii.gz') == []:
            subj_sample = np.random.choice(range(123,156), size=33, replace=True)
            avg = return_average_subjects(subj_sample)
            nib.save(nib.Nifti1Image(avg, affine=init_subj_nib.affine),
                    f'{true_dir}/bootstraps/bootstrap_{b}.nii.gz')

    ##------- load bootstraps
        avg = nib.load(f'{true_dir}/bootstraps/bootstrap_{b}.nii.gz').get_fdata()
        # if i % 10 == 0: print(f"loaded bootstrap_{b}")    
        # get TRxTR
        a = avg[:,:,:,10:] # cut intro TRs
        aTRTR = np.corrcoef(a[overlap].T)
        
    ##------- correlations
        for t in range(nTR):
            # without band
            if TR_band == None:
                corr[i,0,t] = stats.pearsonr(pred[t][notdiag[t]], aTRTR[t][notdiag[t]])[0]
                corr[i,1,t] = stats.pearsonr(lum[t][notdiag[t]], aTRTR[t][notdiag[t]])[0]
            else:
                corr[i,0,t] = stats.pearsonr(pred[t][band[t]], aTRTR[t][band[t]])[0]
                corr[i,1,t] = stats.pearsonr(lum[t][band[t]], aTRTR[t][band[t]])[0]

    # end up with num_bootstraps x 2 
    # make new array for num_bootstraps x num_boundaries x 2
    # then go through each bootstrap and do boundary triggered average for that bootstrap for lum and for pred
    # then overlap them 

    return corr

def get_TR_band(bandwidth, nTR):
    upt = ~np.triu(np.ones(nTR).astype(bool), bandwidth + 1)
    lot = np.triu(np.ones(nTR).astype(bool), -bandwidth)
    notdiag = 1-np.diag(np.ones(nTR)).astype(bool)
    band = upt & lot 
    band = band & notdiag
    band = band.astype(bool)
    return band

def generate_boundary_triggered_averages(corr, skip_other_boundaries=True):
    # Modified apr 29th 2021 for bootstrapping boundaries
    # May 13 changing from 1000x2x21 to 2x21x

    num_bootstraps = corr.shape[0]
    nTR = corr.shape[-1]
    num_boundary_bootstraps = 10
    rand_bounds = np.zeros((num_bootstraps * num_boundary_bootstraps, len(PC_EVENT_BOUNDS[1:])))
    for i in range(num_bootstraps * num_boundary_bootstraps):
        rand_bounds[i,:] = np.random.choice(PC_EVENT_BOUNDS[1:], size=len(PC_EVENT_BOUNDS[1:]),replace=True)
            
    # bound_averages = np.zeros((num_bootstraps * num_boundary_bootstraps,2,21))
    bound_averages = np.full((2,21,num_bootstraps * num_boundary_bootstraps), np.nan)
    window = range(-10, 11)

    for s in tqdm(range(num_bootstraps * num_boundary_bootstraps), desc='Generating bootstrapped boundaries'): 
        brain_idx = s % num_bootstraps
    #     print(brain_idx)
        for i, tr in enumerate(window):
            pred_avg = 0
            lum_avg = 0
            count_bounds = 0
            for b in rand_bounds[s]:
                if b+tr >= nTR:
                    continue
                
                if skip_other_boundaries:
                    # check to see if hitting ±2 TRs from a different boundary
                    hum_bounds_temp = np.array(PC_EVENT_BOUNDS[1:])
                    hum_bounds_temp = np.delete(hum_bounds_temp, np.where(hum_bounds_temp == b)) # ignore our boundary of interest
                    hum_bounds_temp = np.concatenate((hum_bounds_temp-2, hum_bounds_temp-1, 
                                                    hum_bounds_temp, hum_bounds_temp+1,
                                                    hum_bounds_temp+2))
                    if b+tr in hum_bounds_temp:
        #                 print("skipping at", b+tr)
                        continue
                    
                pred_avg += corr[brain_idx,0,b.astype(int)+tr]
                lum_avg  += corr[brain_idx,1,b.astype(int)+tr]
                count_bounds += 1

        #     print(f"count of bounds: {count_bounds}")
            # TODO: Shouldn't need try/excepts here, should be ensuring that every bootstrap
            # has at least one valid TR to analyze
            try:
                pred_avg /= count_bounds
            except:
                pred_avg = 0
            try:
                lum_avg /= count_bounds
            except:
                lum_avg = 0
            bound_averages[0,i,s] = pred_avg
            bound_averages[1,i,s] = lum_avg
    
    return bound_averages


def pc_pred_lum_timecourse(corr):
    num_bootstraps=corr.shape[0]
    nTR = corr.shape[-1]
    plt.figure(figsize=(15, 5))
    plt.tick_params(axis='x', top=False, labeltop=False)
    axes = plt.gca()
    for s in range(num_bootstraps):
        axes.plot(corr[s,0], color='blue', alpha=0.05)
        axes.plot(corr[s,1], color='green', alpha=0.05)

    axes.set_ylim([-1,1])
    axes.set_xlim([0,nTR])
    axes.axhline(0, 0, nTR, linestyle='dashed', color='grey')
    axes.vlines(PC_EVENT_BOUNDS, -.5,.5, linestyle='dashed', color='purple', alpha=0.5, label="event boundaries (hum annotated)")

    h,l = axes.get_legend_handles_labels()
    h.append(Line2D([0], [0], color='green', lw=1))
    h.append(Line2D([0], [0], color='blue', lw=1))
    l.append('corr(lum, true)')
    l.append('corr(pred, true)')
    h.reverse(); l.reverse()
    axes.legend(h,l,fontsize='large',loc='lower right')
    # plt.title("Row by row correlation from TRxTR matrices, all TRs, 100 bootstraps, pred(blue) lum(green)")
    plt.show()

def pc_difference_timecourse(corr):
    num_bootstraps=corr.shape[0]
    nTR = corr.shape[-1]
    plt.figure(figsize=(15, 5))
    plt.tick_params(axis='x', top=False, labeltop=False)
    for s in range(num_bootstraps):
        plt.plot(corr[s,0] - corr[s,1], color='red', alpha=0.05)
        
    axes = plt.gca()
    axes.set_ylim([-1,1])
    axes.set_xlim([0,nTR])
    plt.axhline(0, 0, nTR, linestyle='dashed', color='grey')
    plt.vlines(PC_EVENT_BOUNDS, -.5,.5, linestyle='dashed', color='purple', alpha=0.5, label="event boundaries (hum annotated)")

    h,l = axes.get_legend_handles_labels()
    h.append(Line2D([0], [0], color='red', lw=1))
    l.append('corr(pred, true) - corr(lum, true)')
    h.reverse(); l.reverse()
    axes.legend(h,l,fontsize='large',loc='lower right')
    # plt.title("Row by row correlation from TRxTR matrices, predicted - real, all TRs, 100 bootstraps")
    plt.show()