import img2fmri
from img2fmri import utils
import os, sys, wget, zipfile

ALL_ROIS = ["EarlyVis", "OPA", "LOC", "RSC", "PPA"]
roi_list = ALL_ROIS

input_dir = '/data/commcon/stimuli/frames_by_video/adopt-a-pet_resized_1280-720_frames/'

for roi in ALL_ROIS:
    print(f'Predicting commcon images for {roi}:', file=sys.stderr)
    roi_output = f'data/commcon_{roi}/'
    img2fmri.predict(input_data=input_dir, output_dir=roi_output, roi_list=[roi])
    if roi != ALL_ROIS[-1]: print('\n', file=sys.stderr)
