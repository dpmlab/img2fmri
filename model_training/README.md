### Notebook for training regression models mapping images to fMRI responses

- Uses Pretrained ResNet18 (up to final fully-connected projection layer) for extracting image features
- Uses open source BOLD5000 fMRI data as target data.

File Descriptions:
- model_training.ipynb:
    - Jupyter notebook outlining model training code.
    - Generates a directory 'models' which contains regression models used by img2fmri.
    - NOTE: This requires additional download of BOLD5000 stimulus images and BOLD5000 fMRI data (~1GB space needed). These downloads are executed in the notebook.
- stimuli_list.pkl:
    - fMRI data for list of stimuli, in order they appear, from BOLD5000 project.