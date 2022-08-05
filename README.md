# imgtofmri

**imgtofmri**: a python package for predicting group-level fMRI responses to visual stimuli using deep neural networks

<center>
<img src="model_overview.png", width="700"/>
</center>

Users are encouraged to read the background science information at **THIS LINK**.

## Installation
To install, users must first install [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation) and [AFNI](https://afni.nimh.nih.gov/pub/dist/doc/htmldoc/background_install/main_toc.html), two neuroimaging packages that we use in our pipeline to generate fMRI responses.

Subsequently, users are encouraged to create a new [Conda](https://docs.conda.io/en/latest/) environment with python 3.9, and then can install related dependencies with:

    python -m pip install -r requirements.txt

## Usage
Users are _strongly_ encouraged to view the **overview.ipynb** file, which shows the import and use of the `imgtofmri.predict()` function, as well as its extension to movies using `utils.extract_frames()` and `utils.load_frames()`.

**imgtofmri** can also be used as a command line interface, as:

        python imgtofmri.py [-h] --input input_dir [--output output_dir] [--rois each roi here]
                            [--sigma sigma_val] [--center_crop center_crop]

## Support or Questions
Users are encouraged to email [Max](mailto:mbb2176@columbia.edu) with questions or issues.

## License
This package is licensed under an MIT license in `LICENSE.txt`