import img2fmri
from img2fmri import utils
import os, sys, wget, zipfile

ALL_ROIS = ["EarlyVis", "OPA", "LOC", "RSC", "PPA"]
roi_list = ALL_ROIS

frames_dir = '/data/commcon/stimuli/frames_by_video/'
fits_dir = '/data/commcon/stimuli/output_by_video/'

for dirpath, dirnames, filenames  in os.walk(frames_dir): 
    for ifolder in dirnames:
        print("\nworking on {}".format(ifolder))
        input_path = os.path.join(frames_dir, ifolder)
        output_path= os.path.join(fits_dir, ifolder)
        for roi in ALL_ROIS:
            print(f'\tPredicting commcon images for {roi}:', file=sys.stderr)
            roi_output = os.path.join(output_path, '{}'.format(roi))
            #print('\tinput_path is {}'.format(input_path))
            #print('\troi_output is {}'.format(roi_output))
            img2fmri.predict(input_data=input_path, output_dir=roi_output, roi_list=[roi])
            if roi != ALL_ROIS[-1]: print('\n', file=sys.stderr)
