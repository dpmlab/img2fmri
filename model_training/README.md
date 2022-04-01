### Notebook for training regression models mapping images to brains

- Uses Pretrained ResNET18, up to second to last layer, for extracting image features
- Uses open source BOLD5000 fMRI data, preprocessed (see proprocessing folder for more info), as target data

NOTE: Requires downloads of BOLD5000 images and preprocessed BOLD5000 data:

Presented_Stimuli folder: Pulled from: https://www.dropbox.com/s/5ie18t4rjjvsl47/BOLD5000_Stimuli.zip?dl=0
BOLD5000_Stimuli > Scene_Stimuli > Presented_Stimuli