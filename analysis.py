"""
Analysis and validation functions for imgtofmri python package tutorial.
Tutorial and analysis found at: https://github.com/dpmlab/imgtofmri
Author: Maxwell Bennett mbb2176@columbia.edu
"""
import sys
import glob
import string
import pickle

from natsort import natsorted
from PIL import Image
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import PatchCollection
import pandas as pd
from scipy import stats

from torchvision import transforms
from torch.autograd import Variable

import nibabel as nib

from utils import get_subj_overlap, conv_hrf_and_downsample

ALL_ROIS = ["EarlyVis", "OPA", "LOC", "RSC", "PPA"]
TWINSET_CATEGORIES = ['animals', 'objects', 'scenes', 'people', 'faces']
PC_EVENT_BOUNDS = [0, 12, 26, 44, 56, 71, 81, 92, 104, 119, 123, 136, 143, 151]
TQDM_FORMAT = '{l_bar}{bar:10}{r_bar}{bar:-10b}'


##### Functions for Twinset Analyses:
### Group-level analysis:
def twinset_group_permutations(fmri_dir, n_shuffle=10000, print_vals=False, force_resample=False):
    """
    Generates shuffled permutations for group-level analysis for Twinset.
    print_vals -- will print significance values
    force_resample -- re-generates correlations before calculating null distribution
    """
    if force_resample:
        twinset_group_correlations(fmri_dir)

    corr = np.zeros((156, 156))
    filename = f"data/twinset/correlations/corr_avg_brains.pkl"
    corr = np.load(filename, allow_pickle = True)

    # Remove the duplicate image from training set from this analysis
    corr = np.delete(corr, 19, axis=0)
    corr = np.delete(corr, 19, axis=1)

    df = corr
    num_resample = n_shuffle
    offdiag = (1-np.diag(np.ones(155))).astype(bool)
    real_diff = np.diag(df).mean() - df[offdiag].mean()

    resampled = np.zeros((num_resample, 2))
    shuffled_diffs = np.zeros(num_resample)
    count_over_real = 0
    desc='Performing permutation analysis'
    for i in tqdm(range(num_resample), bar_format=TQDM_FORMAT, desc=desc):
        df = pd.DataFrame(corr)
        df = df.sample(frac=1).reset_index(drop=True)
        df = pd.DataFrame.to_numpy(df)

        resampled[i, 0] = np.diag(df).mean()
        resampled[i, 1] = df[offdiag].mean()
        shuffled_diffs[i] = np.diag(df).mean() - df[offdiag].mean()
        if np.diag(df).mean() - df[offdiag].mean() >= real_diff:
            count_over_real += 1

    p = (count_over_real + 1)/(num_resample + 1)
    twinset_group_plotting(real_diff, shuffled_diffs, p, print_vals)


def twinset_group_correlations(fmri_dir):
    """
    Generates correlations between all predicted and group-averaged brains.
    Called from twinset_group_permutations() if correlations don't exist or if resampling.
    """
    corr = np.zeros((156, 156))
    overlap = get_subj_overlap()

    desc = 'Generating correlations with group-averaged brains'
    for i_pred, image_pred in enumerate(tqdm(range(1,157), bar_format=TQDM_FORMAT, desc=desc)):
        pred_brain = nib.load(f'{fmri_dir}/{"{:03d}".format(image_pred)}.nii.gz').get_fdata()
        for i_test, image_test in enumerate(range(1,157)):
            filename = f'data/twinset/avgs/avg_beta_{"{:04d}".format(image_test)}.nii.gz'
            true_brain = nib.load(filename).get_fdata()
            pred_overlap = pred_brain[overlap]
            true_overlap = true_brain[overlap]

            pearson_r = stats.pearsonr(pred_overlap, true_overlap)[0]
            corr[i_pred, i_test] = pearson_r

    pkl_filename = f"data/twinset/correlations/corr_avg_brains.pkl"
    pickle.dump(corr, open(pkl_filename, 'wb'))


def twinset_group_plotting(real_diff, shuffled_diffs, p, print_vals):
    """ Plots twinset group level analysis """
    fig, ax = plt.subplots(figsize=(7,7))
    ax.scatter(1, real_diff, marker='o', color='k', edgecolors='k', label='Real difference')
    plotparts = ax.violinplot(shuffled_diffs, showextrema=False,)
    plotparts['bodies'][0].set_edgecolor('cornflowerblue')
    plotparts['bodies'][0].set_facecolor('cornflowerblue')
    plotparts['bodies'][0].set_alpha(.7)
    plotparts['bodies'][0].set_label("Null distribution")
    ax.set_xticks([1])
    ax.set_xlim(0.5,1.5)
    ax.set_xticklabels(["Group average"], fontsize=12)
    ax.set_ylabel("Diagonal - off-diagonal", fontsize=12)
    ax.set_title("Difference between diagonal and off-diagonal \n " \
                 "for correlation matrix of predicted and real brains")
    max_y = max(shuffled_diffs.max(), real_diff)
    plot_significance_asterisk(ax.get_xticks()[0], p, max_y, fs=16)
    ax.legend(loc="lower right", fontsize=12)

    if print_vals:
        print(f"Real difference:      {real_diff:.8f}")
        print(f"Mean null difference: {np.mean(shuffled_diffs):.8f}")
        print(f"p value:              {p:.8f}")


### Category level analysis:
def twinset_category_permutations(fmri_dir, 
                                  n_shuffle=1000, 
                                  print_vals=False, 
                                  force_resample=False):
    """ 
    Generates shuffled permutations for category-level analysis for Twinset. 
    print_vals -- will print significance values
    force_resample -- re-generates correlations before calculating null distribution
    """
    if force_resample:
        twinset_category_correlations(fmri_dir)

    filename = f"data/twinset/correlations/corr_categories.pkl"
    full_corr = np.load(filename, allow_pickle=True)
    num_cat = full_corr.shape[0]
    num_resample = n_shuffle
    shuffled_diffs = np.empty((num_cat,num_resample,), dtype=float)
    real_diffs = np.empty((num_cat,), dtype=object)
    p_stats = np.empty((num_cat,2))

    for c in range(num_cat):
        corr = full_corr[c]
        # Remove the one duplicated image from training set, in first category
        if c == 0:
            corr = np.delete(corr, 19, axis=0)
            corr = np.delete(corr, 19, axis=1)
            
        # Calculate difference in unshuffled matrix 
        df = corr # get specific corr matrix for category
        offdiag = (1-np.diag(np.ones(df.shape[0]))).astype(bool)
        real_difference = np.diag(df).mean() - df[offdiag].mean()
        real_diffs[c] = real_difference

        # Generate shuffled permutation matrices
        resampled = np.zeros((num_resample, 2))
        category_diffs = np.zeros(num_resample)
        count_over_real = 0
        desc = f'Generating permutations for \'{TWINSET_CATEGORIES[c]}\''
        for i in tqdm(range(num_resample), bar_format=TQDM_FORMAT, desc=desc):
            df = pd.DataFrame(corr) # reset dataframe
            df = df.sample(frac=1).reset_index(drop=True) # take random sample
            df = pd.DataFrame.to_numpy(df)

            resampled[i, 0] = np.diag(df).mean()
            resampled[i, 1] = df[offdiag].mean()
            category_diffs[i] = np.diag(df).mean() - df[offdiag].mean()
            if np.diag(df).mean() - df[offdiag].mean() >= real_difference:
                count_over_real += 1

        p_stats[c,0] = (count_over_real + 1)/(num_resample + 1)
        shuffled_diffs[c] = category_diffs

    twinset_category_plotting(real_diffs, shuffled_diffs, p_stats, print_vals)


def twinset_category_correlations(fmri_dir):
    """
    Generates correlations between all predicted and group-averaged brains for each image category.
    Called from twinset_category_permutations() if correlations don't exist or if resampling.
    """
    corr = np.zeros((5,), dtype=object)
    overlap = get_subj_overlap()

    for cat_idx, cat_range in enumerate([range(1,29), range(29,65), range(65,101),
                                         range(101, 125), range(125,157)]):
        cat_length = cat_range[-1] - cat_range[0] + 1
        cat_corr = np.zeros((cat_length, cat_length))
        desc = f'Generating correlations for \'{TWINSET_CATEGORIES[cat_idx]}\''
        for i_pred, image_pred in enumerate(tqdm(cat_range, bar_format=TQDM_FORMAT, desc=desc)):
            pred_brain = nib.load(f'{fmri_dir}/{"{:03d}".format(image_pred)}.nii.gz').get_fdata()

            for i_test, image_test in enumerate(cat_range):
                filename = f'data/twinset/avgs/avg_beta_{"{:04d}".format(image_test)}.nii.gz'
                true_brain = nib.load(filename).get_fdata()
                pred_overlap = pred_brain[overlap]
                true_overlap = true_brain[overlap]

                pearson_r = stats.pearsonr(pred_overlap, true_overlap)[0] #just get r, not p val
                cat_corr[i_pred, i_test] = pearson_r
        corr[cat_idx] = cat_corr
        
    print(f"Completed generating category correlations. \n", file=sys.stderr)
    pkl_filename = f"data/twinset/correlations/corr_categories.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(corr, file)


def twinset_category_plotting(real_diffs, shuffled_diffs, p_stats, print_vals):
    """ Plots twinset category-level analysis """
    num_cat = p_stats.shape[0]
    categories = ['animals', 'objects', 'scenes', 'people', 'faces']
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    fig, ax = plt.subplots(figsize=(7,7))
    plotparts = ax.violinplot(shuffled_diffs.T, showextrema = False)
    for i, p in enumerate(plotparts['bodies']):
        p.set_edgecolor(colors[i])
        p.set_facecolor(colors[i])
        p.set_alpha(.5)

    ax.scatter(range(1,6), real_diffs, marker='o', color='k', edgecolors='k', label='Real matrix')
    ax.set_title("Difference between diagonal and off-diagonal \n" \
                 " for correlation matrix of predicted and real brains")
    ax.set_xlabel("Image category", fontsize=12)
    ax.set_ylabel("Diagonal - off-diagonal", fontsize=12)
    ax.set_xticks(range(1,6))
    ax.set_xticklabels(categories, fontsize=12)
    for i, xtick, p in zip(range(num_cat), ax.get_xticks(), p_stats[:,0]):
        max_y = max(real_diffs[i].max(), shuffled_diffs[i].max())
        plot_significance_asterisk(xtick, p, max_y, fs=16)

    # add the multicolor category patch to legend
    h, l = ax.get_legend_handles_labels()
    h.append(MulticolorPatch(colors, alpha=0.5))
    l.append("Null distribution")
    ax.legend(h, l, loc="lower left", fontsize=12,
             handler_map={MulticolorPatch: MulticolorPatchHandler()})
    
    if print_vals:
        for c in range(num_cat):
            print(f'Category: {TWINSET_CATEGORIES[c]}')
            print(f"Real difference:      {real_diffs[c]:.8f}")
            print(f"Mean null difference: {np.mean(shuffled_diffs[c]):.8f}")
            print(f"p value:              {p_stats[c,0]:.8f}\n")


### Participant level analysis:
def twinset_participant_permutations(fmri_dir,
                                     n_shuffle=1000,
                                     print_vals=False,
                                     force_resample=False):
    """
    Generates shuffled permutations for participant-level analysis for Twinset. 
    print_vals -- will print significance values
    force_resample -- re-generates correlations before calculating null distribution
    """
    if force_resample:
        twinset_participant_correlations(fmri_dir)

    filename = f"data/twinset/correlations/corr_per_subj.pkl"
    full_corr = np.load(filename, allow_pickle=True)
    num_cat = full_corr.shape[0]

    num_resample = n_shuffle
    shuffled_diffs = np.empty((num_cat,num_resample,), dtype=float)
    real_diffs = np.empty((num_cat,), dtype=object)
    p_stats = np.empty((num_cat,2))

    desc="Generating permutations for each participant"
    for c in tqdm(range(num_cat), bar_format=TQDM_FORMAT, desc=desc):
        corr = full_corr[c] # get specific corr matrix for subject
        corr = np.delete(corr, 19, axis=0)
        corr = np.delete(corr, 19, axis=1)
        
        df = corr
        offdiag = (1-np.diag(np.ones(df.shape[0]))).astype(bool)
        real_difference = np.diag(df).mean() - df[offdiag].mean()
        real_diffs[c] = real_difference
        resampled = np.zeros((num_resample, 2))
        participant_diffs = np.zeros(num_resample)
        count_over_real = 0
        for i in range(num_resample):
            df = pd.DataFrame(corr) # reset dataframe
            df = df.sample(frac=1).reset_index(drop=True) # take random sample
            df = pd.DataFrame.to_numpy(df)

            resampled[i, 0] = np.diag(df).mean()
            resampled[i, 1] = df[offdiag].mean()
            participant_diffs[i] = np.diag(df).mean() - df[offdiag].mean()
            if np.diag(df).mean() - df[offdiag].mean() >= real_difference:
                count_over_real += 1

        p = (count_over_real + 1)/(num_resample + 1)
        shuffled_diffs[c] = participant_diffs
        p_stats[c,0] = p
    
    twinset_participant_plotting(real_diffs, shuffled_diffs, p_stats, print_vals)


def twinset_participant_correlations(fmri_dir):
    """
    Generates correlations between all predicted and real brains for each participant.
    Called from twinset_participant_permutations() if correlations don't exist or if resampling.
    """
    corr = np.zeros((15,), dtype=object)
    overlap = get_subj_overlap()
    num_images = 156

    total_num_images = 15 * num_images
    desc='Generating correlations per participant'
    with tqdm(total=total_num_images, bar_format=TQDM_FORMAT, desc=desc) as pbar:
        for subj in range(0,15):
            subj_corr = np.zeros((num_images, num_images))

            for i_pred, image_pred in enumerate(range(1,num_images+1)):
                pbar.update(1)
                # if i_pred % 20 == 0: print(f"completed {i_pred} images")
                pred_filename = f'{fmri_dir}/{"{:03d}".format(image_pred)}.nii.gz'
                pred_brain = nib.load(pred_filename).get_fdata()
                for i_test, image_test in enumerate(range(1,num_images+1)):
                    filename = f'data/twinset/subj{"{:02d}".format(subj+1)}/' \
                               f'r_m_s_beta_{"{:04d}".format(image_test)}.nii.gz'
                    true_brain = nib.load(filename).get_fdata()
                    pred_overlap = pred_brain[overlap]
                    true_overlap = true_brain[overlap]

                    pearson_r = stats.pearsonr(pred_overlap, true_overlap)[0] #just get r, not p val
                    subj_corr[i_pred, i_test] = pearson_r

            corr[subj] = subj_corr

    pkl_filename = f"data/twinset/correlations/corr_per_subj.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(corr, file)


def twinset_participant_plotting(real_diffs, shuffled_diffs, p_stats, print_vals):
    """ Plots twinset participant-level analysis """
    num_cat = p_stats.shape[0]
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
    ax.set_title("Difference between diagonal and off-diagonal \n"\
                " for correlation matrix of predicted and real brains")

    same_height = True # changes whether or not to display significance asterisks at same height
    for i, xtick, p in zip(range(num_cat), ax.get_xticks(), p_stats[:,0]):
        if same_height:
            max_y = max(real_diffs[:].max(), shuffled_diffs[:].max())
        else:
            max_y = max(real_diffs[i].max(), shuffled_diffs[i].max())
        plot_significance_asterisk(xtick, p, max_y, fs=12)

    if print_vals:
        for c in range(num_cat):
            print(f"Real difference:      {real_diffs[c]:.10}")
            print(f"Mean null difference: {np.mean(shuffled_diffs[c]):.10f}")
            print(f"p value:              {p_stats[c,0]:.10f}\n")



##### Partly Cloudy Analyses
def get_luminance(input_dir, TRs=168, center_crop=False):
    """
    Calculates the luminance of an image or set of images, and returns a 2-dim object
    with luminance_value x number of frames. Expects an input_dir of image frames
    center_crop -- specifies whether each image is resized or 'squished' to a square, vs. cropped.
    """
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

    mov = []
    sorted_files = natsorted(glob.glob(f'{input_dir}/*'))
    for i, filename in enumerate(tqdm(sorted_files, bar_format=TQDM_FORMAT, desc="Loading images")):
        img = Image.open(filename)
        t_img = Variable(normalize(to_tensor(scaler(img))))
        image = t_img.data.numpy()
        image = np.transpose(image, (1,2,0))

        mov.append(image)

    mov = np.asarray(mov)
    downsampled_shape = np.concatenate((mov.T.shape[0:3], [TRs]))
    mov_convd = np.zeros(downsampled_shape)
    mov_flattened = mov.T.reshape((-1, mov.T.shape[-1]))
    conv_flattened = conv_hrf_and_downsample(mov_flattened, 2, TRs)
    mov_convd = conv_flattened.reshape((downsampled_shape))
    mov_l = mov_convd.T

    # calculate luminance based on RGB values, rescale to 0-255
    y = mov_l[:,:,:,0]*0.2126 + mov_l[:,:,:,1]*0.7152 + mov_l[:,:,:,2]*0.0722
    y = y[2:-8] # trim credits to match real response
    y_flat = y.reshape((y.shape[0], -1))
    y_flat -= np.min(y_flat)
    y_flat /= np.max(y_flat)
    y_flat *= 255

    return y_flat


def generate_bootstrapped_correlations(pred,
                                       true,
                                       lum,
                                       TR_band=None,
                                       num_bootstraps=100,
                                       force_resample=False):
    """ 
    Generates correlations between predicted and real brains, and with luminance and real brains.
    Uses 100 bootstrapped averages of real brains for analyses, and generates those if don't exist
    or if force_resample==True.
    """
    if TR_band != None:
        band = get_TR_band(TR_band, nTR=true.shape[0])

    init_subj_TRs = f'derivatives/init_brain.nii.gz'
    init_subj_nib = nib.load(init_subj_TRs)
    overlap = get_subj_overlap()

    nTR = lum.shape[0]
    notdiag = 1-np.diag(np.ones(158)).astype(bool)
    notdiag = notdiag.astype(bool)
    corr = np.zeros((num_bootstraps, 2, nTR)) # (num_bootstraps x 2 correlations (pred,lum) x TR)

    desc='Generating and loading bootstraps'
    for i, b in enumerate(tqdm(range(num_bootstraps), bar_format=TQDM_FORMAT, desc=desc)):
        # Create bootstraps of real brains if don't exist, or if force_resample==True
        if (glob.glob(f'data/partly_cloudy/bootstraps/bootstrap_{b}.nii.gz') == [] or
            force_resample):
            subj_sample = np.random.choice(range(123,156), size=33, replace=True)
            avg = average_subject_permutation(subj_sample)
            nib.save(nib.Nifti1Image(avg, affine=init_subj_nib.affine),
                    f'data/partly_cloudy/bootstraps/bootstrap_{b}.nii.gz')

        # Load bootstraps
        avg = nib.load(f'data/partly_cloudy/bootstraps/bootstrap_{b}.nii.gz').get_fdata()
        a = avg[:,:,:,10:] # cut intro TRs
        real = np.corrcoef(a[overlap].T)

        # Correlations
        for t in range(nTR):
            if TR_band == None: # Without band around diagonal, aka all non-diagonal TRs
                corr[i,0,t] = stats.pearsonr(pred[t][notdiag[t]], real[t][notdiag[t]])[0]
                corr[i,1,t] = stats.pearsonr(lum[t][notdiag[t]], real[t][notdiag[t]])[0]
            else:
                corr[i,0,t] = stats.pearsonr(pred[t][band[t]], real[t][band[t]])[0]
                corr[i,1,t] = stats.pearsonr(lum[t][band[t]], real[t][band[t]])[0]
    return corr


def pc_pred_lum_timecourse(corr):
    """ Plots the correlations with predicted brains and with luminance model for movie """
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
    axes.vlines(PC_EVENT_BOUNDS, -.5,.5, linestyle='dashed', color='purple', alpha=0.5,
                label="event boundaries")

    h,l = axes.get_legend_handles_labels()
    h.append(Line2D([0], [0], color='green', lw=1))
    h.append(Line2D([0], [0], color='blue', lw=1))
    l.append('corr(luminance, real)')
    l.append('corr(predicted, real)')
    h.reverse(); l.reverse() # corrects ordering of legend and handles
    axes.legend(h,l,fontsize='large',loc='lower right')
    plt.title("Similarity between timepoint-timepoint correlation matrices \n"
              "for predicted response with real response, and luminance model with real response",
              fontsize=14)
    # plt.title("Event structure similarity from correlation matrices " \
    #           "with bootstrapped sampling of real fMRI responses", fontsize=14)
    # plt.title("Row by row correlation of correlation matrices " \
    #           "with bootstrapped sampling of real fMRI responses", fontsize=14)
    axes.set_xlabel("Movie timecourse (TRs, TR=2s)", fontsize=12)
    plt.show()


def pc_difference_timecourse(corr):
    """ Plots the difference in correlations btw predicted brains and luminance model for movie """
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
    plt.vlines(PC_EVENT_BOUNDS, -.5,.5, linestyle='dashed', color='purple', alpha=0.5,
               label="event boundaries")

    h,l = axes.get_legend_handles_labels()
    h.append(Line2D([0], [0], color='red', lw=1))
    l.append('corr(predicted, real) - corr(luminance, real)')
    h.reverse(); l.reverse() # corrects ordering of legend and handles
    axes.legend(h,l,fontsize='large',loc='lower right')
    axes.set_xlabel("Movie timecourse (TRs, TR=2s)", fontsize=12)
    plt.title("Difference between timepoint-timepoint correlation matrix similarities \n"
              "for predicted response with real response, and luminance model with real response",
              fontsize=14)
    # plt.title("Difference between event structure similarity from correlation matrices" \
    #         "with bootstrapped sampling of real fMRI responses", fontsize=14)
    # plt.title("Difference between row by row correlations of correlation matrices " \
    #         "with bootstrapped sampling of real fMRI responses", fontsize=14)
    plt.show()


def generate_boundary_triggered_averages(corr,
                                         num_boundary_bootstraps=10,
                                         skip_other_boundaries=True):
    """
    Generates boundary triggered average analysis by randomly permuting human annotated boundaries
    and averaging the correlations around those boundaries. By default, skips TRs that are ±2 TRs
    from another boundary.
    """
    num_bootstraps = corr.shape[0]
    nTR = corr.shape[-1]
    total_num_bootstraps = num_bootstraps * num_boundary_bootstraps
    rand_bounds = np.zeros((total_num_bootstraps, len(PC_EVENT_BOUNDS[1:])))
    for i in range(total_num_bootstraps):
        rand_bounds[i,:] = np.random.choice(PC_EVENT_BOUNDS[1:],
                                            size=len(PC_EVENT_BOUNDS[1:]),
                                            replace=True)

    bound_averages = np.full((2, 21, total_num_bootstraps), np.nan)
    window = range(-10, 11)

    description = 'Generating bootstrapped boundaries'
    for s in tqdm(range(total_num_bootstraps), bar_format=TQDM_FORMAT, desc=description):
        brain_idx = s % num_bootstraps
        for i, tr in enumerate(window):
            pred_avg = 0
            lum_avg = 0
            count_bounds = 0
            for b in rand_bounds[s]:
                if b+tr >= nTR:
                    continue

                # check to see if hitting ±2 TRs from a different boundary?
                if skip_other_boundaries:
                    hum_bounds_temp = np.array(PC_EVENT_BOUNDS[1:])
                    
                    # ignore our boundary of interest
                    hum_bounds_temp = np.delete(hum_bounds_temp, np.where(hum_bounds_temp == b))
                    hum_bounds_temp = np.concatenate((hum_bounds_temp-2, 
                                                      hum_bounds_temp-1,
                                                      hum_bounds_temp, 
                                                      hum_bounds_temp+1,
                                                      hum_bounds_temp+2))
                    if b+tr in hum_bounds_temp:
                        continue

                pred_avg += corr[brain_idx,0,b.astype(int)+tr]
                lum_avg  += corr[brain_idx,1,b.astype(int)+tr]
                count_bounds += 1

            # Catch an extremely rare case where the permutation of boundaries includes no valid
            # TRs given proximity to other boundaries. Sets value to nan, and eventually to mean.
            try: pred_avg /= count_bounds
            except: pred_avg = np.nan

            try: lum_avg /= count_bounds
            except: lum_avg = np.nan

            bound_averages[0,i,s] = pred_avg
            bound_averages[1,i,s] = lum_avg

    # Resolves rare case explained above
    num_nans = np.argwhere(np.isnan(bound_averages)).shape[0] // 2 
    for _, b, c in np.argwhere(np.isnan(bound_averages))[:num_nans]:
        bound_averages[0,b,c] = np.nanmean(bound_averages[0,b,:])
        bound_averages[1,b,c] = np.nanmean(bound_averages[1,b,:])

    return bound_averages


def pc_bootstrapped_pred_lum_4TRs(bound_averages):
    """ Plots pred and lum correlations for boundary triggered average analysis for ±4TRs """
    nTR = bound_averages.shape[1]
    fig, ax = plt.subplots(figsize=(7,5))
    plt.tick_params(axis='x', top=False, labeltop=False)

    ax.plot(bound_averages[0,6:15], color='blue', alpha = 0.008,)
    ax.plot(bound_averages[1,6:15], color='green', alpha = 0.008,)

    ticks = [i for i in range(0, 9)]
    ticklabels = [f'{i:.0f}' for i in range(-4, 5)]
    ax.set_ylim([-0.5, 0.5])
    ax.set_xlim([0, 8])
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels)
    ax.axhline(0, 0, nTR, linestyle='dashed', color='grey')
    ax.vlines(4, -1, 1, linestyle='dashed', color='purple', alpha=1,
              label="event boundary")

    h,l = ax.get_legend_handles_labels()
    h.append(Line2D([0], [0], color='green', lw=1))
    h.append(Line2D([0], [0], color='blue', lw=1))
    l.append('corr(luminance, real)')
    l.append('corr(predicted, real)')
    h.reverse(); l.reverse() # corrects ordering of legend and handles
    ax.legend(h,l,fontsize='medium',loc='lower right')
    plt.title("Boundary triggered average of bootstrapped boundaries \n" \
              "with bootstrapped sampling of real fMRI responses", fontsize=14)
    ax.set_xlabel("TRs surrounding boundary (TR=2s)", fontsize=12)

    plt.show()


def pc_bootstrapped_difference_4TRs(bound_averages):
    """ Plots difference between boundary triggered average correlations for ±4TRs """
    nTR = bound_averages.shape[1]
    fig, ax = plt.subplots(figsize=(7,5))
    plt.tick_params(axis='x', top=False, labeltop=False)

    diffs = bound_averages[0,6:15]-bound_averages[1,6:15]
    ax.plot(diffs, color='red', alpha = 0.008,)

    ticks = [i for i in range(0, 9)]
    ticklabels = [f'{i:.0f}' for i in range(-4, 5)]
    ax.set_ylim([-0.5,0.5])
    ax.set_xlim([0,8])
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels)
    ax.axhline(0, 0, nTR, linestyle='dashed', color='grey')
    ax.vlines(4, -1, 1, linestyle='dashed', color='purple', alpha=1,
              label="event boundary")

    h,l = ax.get_legend_handles_labels()
    h.append(Line2D([0], [0], color='red', lw=1))
    l.append('corr(predicted, real) - corr(luminance, real)')
    h.reverse(); l.reverse() # corrects ordering of legend and handles
    ax.legend(h,l,fontsize='medium',loc='lower right')
    # plt.title("Event structure similarity from correlation matrices " \
    #           "with bootstrapped sampling of real fMRI responses")
    plt.title("Boundary triggered average of difference between correlation "\
              "matrices \n with bootstrapped boundaries " \
              "and sampling of real fMRI responses", fontsize=14)
    ax.set_xlabel("TRs surrounding boundary (TR=2s)", fontsize=12)

    max_y = diffs.max()
    plot_significance_asterisks_bootstraps(num_bootstraps_below_zero(diffs), max_y)
    plt.show()


def get_TR_band(bandwidth, nTR):
    """
    Returns a 1D bool array for indexing into specific bands around diagonal
    """
    upt = ~np.triu(np.ones(nTR).astype(bool), bandwidth + 1)
    lot = np.triu(np.ones(nTR).astype(bool), -bandwidth)
    notdiag = 1-np.diag(np.ones(nTR)).astype(bool)
    band = upt & lot
    band = band & notdiag
    band = band.astype(bool)
    return band


def num_bootstraps_below_zero(diffs):
    """ Returns number of samples below zero for calculating p-values on difference plots """
    len_diffs = diffs.shape[0]
    num_bootstraps = diffs.shape[1]

    num_below_zero = np.zeros((len_diffs))
    for i in range(0, len_diffs):
        for b in range(num_bootstraps):
            if diffs[i,b] < 0:
                num_below_zero[i] += 1

    return (num_below_zero+1) / (num_bootstraps+1)


def average_subject_permutation(subject_list):
    """ Returns avg (zscore of sum of zscored (+ra, +dcto)) of a given sample of real brains """
    init_filename = f'derivatives/init_brain.nii.gz' # For initializing brain
    init_brain_nib = nib.load(init_filename)
    sum_brains = np.full((init_brain_nib.shape), np.nan)
    nTR = 168
    overlap = get_subj_overlap()

    for subj in subject_list:
        subj_TR = f'data/partly_cloudy/subjects_preprocessed/subj{subj}.nii.gz'
        subj_brain = nib.load(subj_TR).get_fdata()
        sum_brains[overlap] = np.nansum([sum_brains[overlap], subj_brain[overlap]], axis=0)

    sum_brains[overlap] = stats.zscore(sum_brains[overlap], axis=-1)
    return sum_brains


##### Plotting functions:
def plot_correlation_matrices(pred, real, lum):
    """ Plots correlation matrices for Partly Cloudy analyses """
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12,4))
    axes[0].imshow(pred, vmin=-1, vmax=1)
    axes[1].imshow(real, vmin=-1, vmax=1)
    im = axes[2].imshow(lum, vmin=-1, vmax=1)
    cbar_ax = fig.add_axes([0.35, -0.05, 0.33, 0.05])
    fig.colorbar(im, cax=cbar_ax, shrink=0.6, orientation='horizontal')
    axes[0].set_xlabel('Predicted brains', fontsize=12)
    axes[1].set_xlabel('Real brains', fontsize=12)
    axes[2].set_xlabel('Image luminace', fontsize=12)

    for a, ax in enumerate(axes.ravel()):
        ax.text(-0.13, -.15, string.ascii_lowercase[a], transform=ax.transAxes, 
                size=16, weight='bold')
    fig.suptitle('Correlation matrices for Partly Cloudy', fontsize=16, y=1)


def plot_preprocessing_matrices(pred_voxels, conv, removed_avg, removed_dct):
    """ Plots correlation matrices for preprocessing steps """
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16,4))
    axes[0].imshow(np.corrcoef(pred_voxels.T), vmin=-1, vmax=1)
    axes[1].imshow(np.corrcoef(conv[:, 2:].T), vmin=-1, vmax=1)
    im = axes[2].imshow(np.corrcoef(removed_avg[:, 2:].T), vmin=-1, vmax=1)
    axes[3].imshow(np.corrcoef(removed_dct[:, 2:].T), vmin=-1, vmax=1)
    cbar_ax = fig.add_axes([0.35, -0.05, 0.33, 0.05])
    fig.colorbar(im, cax=cbar_ax, shrink=0.6, orientation='horizontal')
    axes[0].set_xlabel('Predicted response', fontsize=12)
    axes[1].set_xlabel('Convolve w/ HRF, downsample', fontsize=12)
    axes[2].set_xlabel('Remove average', fontsize=12)
    axes[3].set_xlabel('Remove DCT', fontsize=12)

    for a, ax in enumerate(axes.ravel()):
        ax.text(-0.13, -.15, string.ascii_lowercase[a], transform=ax.transAxes, 
                size=16, weight='bold')
    fig.suptitle('Correlation matrices of Partly Cloudy prediction preprocessing', 
                 fontsize=16, y=.95)


# Asterisk code modified from https://stackoverflow.com/a/52333561
def plot_significance_asterisk(x, data, max_y, maxasterix=None, fs=12):
    """
    Plots significance markers
    † is <0.075,      * is p < 0.05
    ** is p < 0.01,   *** is p < 0.001, etc.
    """
    text = "\u2020" if data < 0.075 and data > 0.05 else ""

    p = .05 
    if data < p:
        text += "*"
    p = 0.01
    while data < p:
        text += '*'
        p /= 10.

        if maxasterix and len(text) == maxasterix:
            break
    
    if text != "":
        y = max_y * 1.05
        ax = plt.gca()
        y_lower, y_upper = ax.get_ylim()
        if y * 1.08 > y_upper:
            ax.set_ylim(y_lower, y_upper * 1.08)
        ax.text(x, y, text, ha='center', fontsize=fs)
        xticks = ax.get_xticks()


def plot_significance_asterisks_bootstraps(data, max_y, maxasterix=None, fs=12):
    """
    Plots significance markers for boundary triggered average analysis
    † is <0.075,      * is p < 0.05
    ** is p < 0.01,   *** is p < 0.001, etc.
    """
    # Get signifiance marker (if any) for each TR
    asterisks = np.empty((len(data)),dtype=object)
    for i, d in enumerate(data):
        text = "\u2020" if d < 0.075 and d > 0.05 else ""
        p = .05 
        if d < p:
            text += "*"
        p = 0.01
        while d < p:
            text += '*'
            p /= 10.
            if maxasterix and len(text) == maxasterix:
                break
        asterisks[i] = text

    # Check if neighboring TRs require same significance marker
    markers = {}
    i = 0
    while i < len(asterisks):
        a = asterisks[i]
        if a == "":
            i += 1
            continue
        if a not in markers:
            markers[a] = list()
        j = i + 1
        same_indices = [ i ]
        while j < len(asterisks):
            i = j
            if asterisks[j] == a:
                same_indices.append(j)
                j += 1
            else:
                i -= 1
                break
        markers[a].append(same_indices)
        i += 1

    # Add significance markers to plot
    for t in markers:
        for l_of_t in markers[t]:
            y = max_y * 1.05
            ax = plt.gca()
            y_lower, y_upper = ax.get_ylim()
            if y * 1.08 > y_upper:
                ax.set_ylim(y_lower, y_upper * 1.08)
            mid_x = (l_of_t[0] + l_of_t[-1]) / 2
            ax.text(mid_x, y, t, ha='center', fontsize=fs)
            xticks = ax.get_xticks()
            ax.axhline(y*0.85, 
                       (l_of_t[0]-0.425)/xticks[-1],
                       (l_of_t[-1]+0.425)/xticks[-1],
                       linestyle='-', color='black')

# Multicolor patch code below assisted from https://stackoverflow.com/a/67870930/12685473
class MulticolorPatch(object):
    """ Define an object that will be used by the legend """
    def __init__(self, colors, alpha=1):
        self.colors = colors
        self.alpha = alpha


class MulticolorPatchHandler(object):
    """ Define a handler for the MulticolorPatch object """
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
