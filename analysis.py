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
TWINSET_CATEGORIES = ['animals', 'objects', 'scenes', 'people', 'faces']
PC_EVENT_BOUNDS = [0, 12, 26, 44, 56, 71, 81, 92, 104, 119, 123, 136, 143, 151]

### Functions for Twinset Analysis:
## Group-level analyses:

# Generates correlations between all predicted and group-averaged brains.
def twinset_generate_group_correlations(image_dir):
    corr = np.zeros((156, 156))
    overlap = get_subj_overlap()

    for i_pred, image_pred in enumerate(tqdm(range(1,157), desc='Generating correlations with '\
                                                                'group-averaged brains')):
        pred_brain = nib.load(f'{image_dir}/{"{:03d}".format(image_pred)}.nii.gz').get_fdata()
        for i_test, image_test in enumerate(range(1,157)):
            filename = f'data/twinset/avgs/avg_beta_{"{:04d}".format(image_test)}.nii.gz'
            true_brain = nib.load(filename).get_fdata()
            pred_overlap = pred_brain[overlap]
            true_overlap = true_brain[overlap]

            pearson_r = stats.pearsonr(pred_overlap, true_overlap)[0]
            corr[i_pred, i_test] = pearson_r

    pkl_filename = f"data/twinset/correlations/corr_avg_brains.pkl"
    pickle.dump(corr, open(pkl_filename, 'wb'))


#
def twinset_random_group_permutations(image_dir, n_shuffle, from_scratch=False, print_vals=False):
    if from_scratch:
        twinset_generate_group_correlations(image_dir)
    
    corr = np.zeros((156, 156))
    filename = f"data/twinset/correlations/corr_avg_brains.pkl"
    corr = np.load(filename, allow_pickle = True)

    df = corr
    num_resample = n_shuffle
    offdiag = (1-np.diag(np.ones(156))).astype(bool)
    real_diff = np.diag(df).mean() - df[offdiag].mean()

    resampled = np.zeros((num_resample, 2))
    diffs = np.zeros(num_resample)
    count_over_real = 0
    for i in tqdm(range(num_resample), desc='Performing permutation analysis'):
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
    ax.set_title("Difference between diagonal and off-diagonal \n " \
                 "for correlation matrix of real and predicted brains")
    max_y = max(diffs.max(), real_diff)

    plot_significance_asterisk(ax.get_xticks()[0], max_y,  p, fs=16)
    ax.legend(loc="lower right", fontsize=12)

#   Category correlations:
def twinset_generate_category_correlations(output_dir):
    corr = np.zeros((5,), dtype=object)
    overlap = get_subj_overlap()

    for cat_idx, cat_range in enumerate([range(1,29), range(29,65), range(65,101),
                                         range(101, 125), range(125,157)]):
        cat_length = cat_range[-1] - cat_range[0] + 1
        cat_corr = np.zeros((cat_length, cat_length))
        tqdm_description = f'Generating correlations for \'{TWINSET_CATEGORIES[cat_idx]}\''
        for i_pred, image_pred in enumerate(tqdm(cat_range, desc=tqdm_description)):
            pred_brain = nib.load(f'{output_dir}/{"{:03d}".format(image_pred)}.nii.gz').get_fdata()
            # if i_pred % 20 == 0: print(f"completed {i_pred} images")

            for i_test, image_test in enumerate(cat_range):
                filename = f'data/twinset/avgs/avg_beta_{"{:04d}".format(image_test)}.nii.gz'
                true_brain = nib.load(filename).get_fdata()
                pred_overlap = pred_brain[overlap]
                true_overlap = true_brain[overlap]

                pearson_r = stats.pearsonr(pred_overlap, true_overlap)[0] #just get r, not p val
                cat_corr[i_pred, i_test] = pearson_r

        corr[cat_idx] = cat_corr
        # print(f"Completed category: {TWINSET_CATEGORIES[cat_idx]}")

    pkl_filename = f"data/twinset/correlations/corr_categories.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(corr, file)

def twinset_random_category_permutations(n_shuffle=1000, print_vals=False):
    filename = f"data/twinset/correlations/corr_categories.pkl"
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
    ax.set_title("Difference between diagonal and off-diagonal \n" \
                 " for correlation matrix of real and predicted brains")
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

            #  TODO no need for enumerates since ranges
            for i_pred, image_pred in enumerate(range(1,num_images+1)):
                pbar.update(1)
                # if i_pred % 20 == 0: print(f"completed {i_pred} images")
                pred_filename = f'{output_dir}/{"{:03d}".format(image_pred)}.nii.gz'
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

def twinset_random_participant_permutations(n_shuffle=1000, print_vals=False):
    filename = f"data/twinset/correlations/corr_per_subj.pkl"
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
    ax.set_title("Difference between diagonal and off-diagonal \n"\
                " for correlation matrix of real and predicted brains")

    same_height = True
    for i, xtick, p in zip(range(num_cat), ax.get_xticks(), pstats[:,0]):
        if same_height:
            max_y = max(real_diffs[:].max(), shuffled_diffs[:].max())
        else:
            max_y = max(real_diffs[i].max(), shuffled_diffs[i].max())

        plot_significance_asterisk(xtick, max_y, p, fs=12)


# TODO modified from
# https://stackoverflow.com/questions/11517986/
# indicating-the-statistically-significant-difference-in-bar-graph
def plot_significance_asterisk(x, max_y, data, maxasterix=None, fs=12):
    if type(data) is str:
        text = data
    else:
        # † (\u2020) is <0.075
        # * is p < 0.05
        # ** is p < 0.01
        # *** is p < 0.001
        # etc.
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
        # print((x-0.49)/xticks[-1], (x+0.5)/xticks[-1])
        # ax.axhline(y*0.95, (x-0.425)/xticks[-1], (x+0.425)/xticks[-1], linestyle='-', color='black')

# TODO cleanup...
def plot_significance_asterisks_bootstraps(data, max_y, maxasterix=None, fs=12):
    # † (\u2020) is <0.075
    # * is p < 0.05
    # ** is p < 0.01
    # *** is p < 0.001
    # etc.
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

    same_ticks = {}
    i = 0
    while i < len(asterisks):
        a = asterisks[i]
        if a == "":
            i += 1
            continue
        if a not in same_ticks:
            same_ticks[a] = list()

        j = i + 1
        same_indices = [ i ]
        while j < len(asterisks):
            i = j
            if asterisks[j] == a:
                same_indices.append(j)
                j += 1
            else: 
                break
        same_ticks[a].append(same_indices)

    for t in same_ticks:
        for l_of_t in same_ticks[t]:
            y = max_y * 1.05
            ax = plt.gca()
            y_lower, y_upper = ax.get_ylim()
            if y * 1.08 > y_upper:
                ax.set_ylim(y_lower, y_upper * 1.08)
            mid_x = (l_of_t[0] + l_of_t[-1]) / 2
            ax.text(mid_x, y, t, ha='center', fontsize=fs)
            xticks = ax.get_xticks()
            ax.axhline(y*0.95, 
                       (l_of_t[0]-0.425)/xticks[-1],
                       (l_of_t[-1]+0.425)/xticks[-1],
                       linestyle='-', color='black')


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


def pc_bootstrapped_pred_lum(bound_averages):
    # plt.figure(figsize=(5, 5))
    nTR = bound_averages.shape[1]
    fig, ax = plt.subplots(figsize=(7,5))
    plt.tick_params(axis='x', top=False, labeltop=False)

    sorted_arr = np.sort(bound_averages, axis=2)
    ax.plot(bound_averages[0], color='blue', alpha = 0.008,)
    ax.plot(bound_averages[1], color='green', alpha = 0.008,)


    ticks = [i for i in range(0, 21)][::2]
    ticklabels = [f'{i:.0f}' for i in range(-10, 11)][::2]
    ax.set_ylim([-1, 1])
    ax.set_xlim([0, 20])
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels)
    ax.axhline(0, 0, nTR, linestyle='dashed', color='grey')
    ax.vlines(10, -1, 1, linestyle='dashed', color='purple', alpha=1)

    # plt.legend(fontsize='large', loc='lower right')
    # fig.suptitle("""Boundary triggered average for bootstrapped brains and boundaries
    # btw corr of TRxTR matrices, all TRs predicted(blue) luminance(green)
    # skipping ±2TRs from bounds\n""", y=1.03, fontsize=14)
    plt.show()

def pc_bootstrapped_pred_lum_3TRs(bound_averages, y_min=-0.5, y_max=0.5):
    nTR = bound_averages.shape[1]
    fig, ax = plt.subplots(figsize=(5,5))
    plt.tick_params(axis='x', top=False, labeltop=False)

    sorted_arr = np.sort(bound_averages, axis=2)
    ax.plot(bound_averages[0,7:14], color='blue', alpha = 0.008,
                label=f'avg corr(pred,true), mean: {np.mean(bound_averages[:,0,:]):.3f}')
    ax.plot(bound_averages[1,7:14], color='green', alpha = 0.008,
                label=f'avg corr(lum,true), mean: {np.mean(bound_averages[:,1,:]):.3f}')


    ticks = [i for i in range(0, 7)]
    ticklabels = [f'{i:.0f}' for i in range(-3, 4)]
    ax.set_ylim([y_min,y_max])
    ax.set_xlim([0,6])
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels)
    ax.axhline(0, 0, nTR, linestyle='dashed', color='grey')
    ax.vlines(3, -1, 1, linestyle='dashed', color='purple', alpha=1)

    # plt.legend(fontsize='large', loc='lower right')
    # fig.suptitle("""Boundary triggered average for bootstrapped brains and boundaries
    # btw corr of TRxTR matrices, all TRs predicted(blue) luminance(green)
    # skipping ±2TRs from bounds\n""", y=1.03, fontsize=14)
    plt.show()

def pc_bootstrapped_difference(bound_averages):
    # plt.figure(figsize=(5, 5))
    nTR = bound_averages.shape[1]
    fig, ax = plt.subplots(figsize=(7,5))
    plt.tick_params(axis='x', top=False, labeltop=False)

    # sorted_arr = np.sort(bound_averages, axis=2)
    diffs = bound_averages[0]-bound_averages[1]
    ax.plot(diffs, color='red', alpha = 0.008,)

    ticks = [i for i in range(0, 21)][::2]
    ticklabels = [f'{i:.0f}' for i in range(-10, 11)][::2]
    ax.set_ylim([-1,1])
    ax.set_xlim([0,20])
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels)
    ax.axhline(0, 0, nTR, linestyle='dashed', color='grey')
    
    max_y = diffs[:].max()
    ax.vlines(10, -1, 1, linestyle='dashed', color='purple', alpha=1)
     
    max_y = diffs.max()
    plot_significance_asterisks_bootstraps(num_bootstraps_below_zero(diffs), max_y)
    # same_height = True
    # for i, p in zip(range(len(diffs)), num_bootstraps_below_zero(diffs)):
    #     if same_height:
    #         max_y = diffs[:].max()
    #     else:
    #         max_y = diffs[i].max()
    #     plot_significance_asterisk(i, max_y, p, fs=14)

    # print(num_below_zero(diffs))
    # print(ax.get_xticks())
    # plt.legend(fontsize='large', loc='lower right')
    # fig.suptitle("""Boundary triggered average for bootstrapped brains and boundaries
    # btw corr of TRxTR matrices, all TRs predicted-luminance
    # skipping ±2TRs from bounds\n""", y=1.03, fontsize=14)
    plt.show()

def pc_bootstrapped_difference_3TRs(bound_averages):
    # plt.figure(figsize=(5, 5))
    nTR = bound_averages.shape[1]
    fig, ax = plt.subplots(figsize=(5,5))
    plt.tick_params(axis='x', top=False, labeltop=False)

    # sorted_arr = np.sort(bound_averages, axis=2)
    diffs = bound_averages[0,7:14]-bound_averages[1,7:14]
    ax.plot(diffs, color='red', alpha = 0.008,)

    ticks = [i for i in range(0, 7)]
    ticklabels = [f'{i:.0f}' for i in range(-3, 4)]
    ax.set_ylim([-0.5,0.5])
    ax.set_xlim([0,6])
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels)
    ax.axhline(0, 0, nTR, linestyle='dashed', color='grey')
    # plt.vlines(PC_EVENT_BOUNDS, -.5,.5, linestyle='dashed', color='red',
    #            alpha=0.5, label="event boundaries (hum annotated)")

    ax.vlines(3, -1, 1, linestyle='dashed', color='purple', alpha=1)

    max_y = diffs.max()
    plot_significance_asterisks_bootstraps(num_bootstraps_below_zero(diffs), max_y)
    # same_height = True
    # for i, p in zip(range(len(diffs)), num_bootstraps_below_zero(diffs)):
    #     if same_height:
    #         max_y = diffs[:].max()
    #     else:
    #         max_y = diffs[i].max()
    #     plot_significance_asterisk(i, max_y, p, fs=14)

    # fig.suptitle("""Boundary triggered average for bootstrapped brains and boundaries
    # btw corr of TRxTR matrices, all TRs predicted-luminance
    # skipping ±2TRs from bounds\n""", y=1.03, fontsize=14)
    plt.show()

def generate_bootstrapped_correlations(true, 
                                       pred, 
                                       lum, 
                                       TR_band=None,
                                       num_bootstraps=100):
    if TR_band != None:
        band = get_TR_band(TR_band, nTR=true.shape[0])

    # LOAD Bootstraps instead of generating
    # set up corr structure for shape: (num_bootstraps, 2 (lum, boot_real; pred, boot_real) x TR)
    init_subj = 123 # For initializing brain
    init_subj_TRs = f'derivatives/init_brain.nii.gz'
    init_subj_nib = nib.load(init_subj_TRs)
    overlap = get_subj_overlap()

    # num_bootstraps = 100
    nTR = lum.shape[0]
    notdiag = 1-np.diag(np.ones(158)).astype(bool)
    notdiag = notdiag.astype(bool)
    corr = np.zeros((num_bootstraps, 2, nTR))

    for i, b in enumerate(tqdm(range(num_bootstraps), desc='Generating and loading bootstraps')):
    #------- save bootstrap brains if don't exist
        if glob.glob(f'data/partly_cloudy/bootstraps/bootstrap_{b}.nii.gz') == []:
            subj_sample = np.random.choice(range(123,156), size=33, replace=True)
            avg = average_subject_permutation(subj_sample)
            nib.save(nib.Nifti1Image(avg, affine=init_subj_nib.affine),
                    f'data/partly_cloudy/bootstraps/bootstrap_{b}.nii.gz')

    ##------- load bootstraps
        avg = nib.load(f'data/partly_cloudy/bootstraps/bootstrap_{b}.nii.gz').get_fdata()
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
    # then go through each bootstrap and do boundary triggered average for that bootstrap for
    # lum and for pred then overlap them

    return corr

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

def generate_boundary_triggered_averages(corr,
                                         num_boundary_bootstraps=10,
                                         skip_other_boundaries=True):
    num_bootstraps = corr.shape[0]
    nTR = corr.shape[-1]
    rand_bounds = np.zeros((num_bootstraps * num_boundary_bootstraps, len(PC_EVENT_BOUNDS[1:])))
    for i in range(num_bootstraps * num_boundary_bootstraps):
        rand_bounds[i,:] = np.random.choice(PC_EVENT_BOUNDS[1:],
                                            size=len(PC_EVENT_BOUNDS[1:]),
                                            replace=True)

    bound_averages = np.full((2,21,num_bootstraps * num_boundary_bootstraps), np.nan)
    window = range(-10, 11)

    description = 'Generating bootstrapped boundaries'
    for s in tqdm(range(num_bootstraps * num_boundary_bootstraps), desc=description):
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
                    hum_bounds_temp = np.concatenate((hum_bounds_temp-2, hum_bounds_temp-1,
                                                    hum_bounds_temp, hum_bounds_temp+1,
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
    axes.vlines(PC_EVENT_BOUNDS, -.5,.5, linestyle='dashed', color='purple', alpha=0.5,
                label="event boundaries (hum annotated)")

    h,l = axes.get_legend_handles_labels()
    h.append(Line2D([0], [0], color='green', lw=1))
    h.append(Line2D([0], [0], color='blue', lw=1))
    l.append('corr(lum, true)')
    l.append('corr(pred, true)')
    h.reverse(); l.reverse()
    axes.legend(h,l,fontsize='large',loc='lower right')
    # plt.title("Row by row correlation from TRxTR matrices, " \
    #           " all TRs, 100 bootstraps, pred(blue) lum(green)")
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
    plt.vlines(PC_EVENT_BOUNDS, -.5,.5, linestyle='dashed', color='purple', alpha=0.5,
               label="event boundaries (hum annotated)")

    h,l = axes.get_legend_handles_labels()
    h.append(Line2D([0], [0], color='red', lw=1))
    l.append('corr(pred, true) - corr(lum, true)')
    h.reverse(); l.reverse()
    axes.legend(h,l,fontsize='large',loc='lower right')
    plt.show()


def pc_bootstrapped_pred_lum_conf_intervals(bound_averages):
    # plt.figure(figsize=(5, 5))
    nTR = bound_averages.shape[1]
    # fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
    fig, ax = plt.subplots(figsize=(7,5))
    plt.tick_params(axis='x', top=False, labeltop=False)

    sorted_arr = np.sort(bound_averages, axis=2)
    # axes[0].plot(bound_averages[0], color='blue', alpha = 0.008,)
    # axes[0].plot(bound_averages[1], color='green', alpha = 0.008,)
    ax.fill_between(range(21), sorted_arr[0,:,50], sorted_arr[0,:,-50], alpha=0.5, color="blue")
    ax.fill_between(range(21), sorted_arr[1,:,50], sorted_arr[1,:,-50], alpha=0.5, color="green")


    ticks = [i for i in range(0, 21)][::2]
    ticklabels = [f'{i:.0f}' for i in range(-10, 11)][::2]
    ax.set_ylim([-1,1])
    ax.set_xlim([0,20])
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels)
    ax.axhline(0, 0, nTR, linestyle='dashed', color='grey')
    ax.vlines(10, -1, 1, linestyle='dashed', color='purple', alpha=1)

    # plt.legend(fontsize='large', loc='lower right')
    # fig.suptitle("""Boundary triggered average for bootstrapped brains and boundaries
    # btw corr of TRxTR matrices, all TRs predicted(blue) luminance(green)
    # skipping ±2TRs from bounds\n""", y=1.03, fontsize=14)
    plt.show()

def pc_bootstrapped_pred_lum_3TRs_conf_intervals(bound_averages):
    nTR = bound_averages.shape[1]
    fig, ax = plt.subplots(figsize=(5,5))
    plt.tick_params(axis='x', top=False, labeltop=False)

    sorted_arr = np.sort(bound_averages, axis=2)
    ax.fill_between(range(7), sorted_arr[0,7:14,50], sorted_arr[0,7:14,-50], alpha=0.5, color="b")
    ax.fill_between(range(7), sorted_arr[1,7:14,50], sorted_arr[1,7:14,-50], alpha=0.5, color="g")


    ticks = [i for i in range(0, 7)]
    ticklabels = [f'{i:.0f}' for i in range(-3, 4)]
    ax.set_ylim([-0.5,1])
    ax.set_xlim([0,6])
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels)
    ax.axhline(0, 0, nTR, linestyle='dashed', color='grey')
    ax.vlines(3, -1, 1, linestyle='dashed', color='purple', alpha=1)

    # plt.legend(fontsize='large', loc='lower right')
    # fig.suptitle("""Boundary triggered average for bootstrapped brains and boundaries
    # btw corr of TRxTR matrices, all TRs predicted(blue) luminance(green)
    # skipping ±2TRs from bounds\n""", y=1.03, fontsize=14)
    plt.show()

def pc_bootstrapped_difference_conf_intervals(bound_averages):
    # plt.figure(figsize=(5, 5))
    nTR = bound_averages.shape[1]
    fig, ax = plt.subplots(figsize=(7,5))
    plt.tick_params(axis='x', top=False, labeltop=False)

    sorted_arr = np.sort(bound_averages, axis=2)
    ax.fill_between(range(21),
                    sorted_arr[0,:,50]-sorted_arr[1,:,50],
                    sorted_arr[0,:,-50]-sorted_arr[1,:,-50], alpha=0.5, color="red")

    ticks = [i for i in range(0, 21)][::2]
    ticklabels = [f'{i:.0f}' for i in range(-10, 11)][::2]
    ax.set_ylim([-1,1])
    ax.set_xlim([0,20])
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels)
    ax.axhline(0, 0, nTR, linestyle='dashed', color='grey')
    ax.vlines(10, -1, 1, linestyle='dashed', color='purple', alpha=1)

    # plt.legend(fontsize='large', loc='lower right')
    # fig.suptitle("""Boundary triggered average for bootstrapped brains and boundaries
    # btw corr of TRxTR matrices, all TRs predicted-luminance
    # skipping ±2TRs from bounds\n""", y=1.03, fontsize=14)
    plt.show()

def pc_bootstrapped_difference_3TRs_conf_intervals(bound_averages):
    # plt.figure(figsize=(5, 5))
    nTR = bound_averages.shape[1]
    fig, ax = plt.subplots(figsize=(5,5))
    plt.tick_params(axis='x', top=False, labeltop=False)

    sorted_arr = np.sort(bound_averages, axis=2)
    ax.fill_between(range(7),
                    sorted_arr[0,7:14,50]-sorted_arr[1,7:14,50],
                    sorted_arr[0,7:14,-50]-sorted_arr[1,7:14,-50], alpha=0.5, color="red")

    ticks = [i for i in range(0, 7)]
    ticklabels = [f'{i:.0f}' for i in range(-3, 4)]
    ax.set_ylim([-0.5,0.5])
    ax.set_xlim([0,6])
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels)
    ax.axhline(0, 0, nTR, linestyle='dashed', color='grey')
    # plt.vlines(PC_EVENT_BOUNDS, -.5,.5, linestyle='dashed', color='red', alpha=0.5,
    #            label="event boundaries (hum annotated)")
    ax.vlines(3, -1, 1, linestyle='dashed', color='purple', alpha=1)

    # fig.suptitle("""Boundary triggered average for bootstrapped brains and boundaries
    # btw corr of TRxTR matrices, all TRs predicted-luminance
    # skipping ±2TRs from bounds\n""", y=1.03, fontsize=14)
    plt.show()

def num_bootstraps_below_zero(diffs):
    len_diffs = diffs.shape[0]
    num_bootstraps = diffs.shape[1]

    num_below_zero = np.zeros((len_diffs))
    for i in range(0, len_diffs):
        for b in range(num_bootstraps):
            if diffs[i,b] < 0:
                num_below_zero[i] += 1

    return (num_below_zero+1) / (num_bootstraps+1)

# returns zscore of sum of zscored (+ra, +dcto) individual real brains
def average_subject_permutation(subject_list):
    # print("Subject_list: ", subject_list)
    init_subj = subject_list[0] # For initializing brain
    init_subj_TRs = f'derivatives/init_brain.nii.gz'
    init_subj_nib = nib.load(init_subj_TRs)
    sum_brains = np.full((init_subj_nib.shape), np.nan)

    nTR = 168
    overlap = get_subj_overlap()
    for subj in subject_list:
        subj_TR = f'data/partly_cloudy/subjects_preprocessed/subj{subj}.nii.gz'
        subj_brain = nib.load(subj_TR).get_fdata()
        # sum_brains = np.nansum([sum_brains, subj_brain], axis=0)
        sum_brains[overlap] = np.nansum([sum_brains[overlap], subj_brain[overlap]], axis=0)

    sum_brains[overlap] = stats.zscore(sum_brains[overlap], axis=-1)
    return sum_brains