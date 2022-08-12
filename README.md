# imgtofmri

**imgtofmri**: a python package for predicting group-level fMRI responses to visual stimuli using deep neural networks

<center>
<img src="model_overview.png", width="700"/>
</center>

Users are encouraged to read the background science information for an overview of the model and its intended uses ([science_overview.pdf](science_overview.pdf)).

## Installation
To install, users must first install [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation) and [AFNI](https://afni.nimh.nih.gov/pub/dist/doc/htmldoc/background_install/main_toc.html), two neuroimaging packages that we use in our pipeline to generate fMRI responses.

After cloning this repository, users should install [Conda](https://docs.conda.io/en/latest/) and create an environment with python 3.9, e.g.: 

    conda create --name name_of_environment python=3.9
    conda activate name_of_environment

Users can change directories to this repository, then install package dependencies with:

    python -m pip install -r requirements.txt

Users are now able to use the command-line interface for predicting fMRI responses as outlined in the section below.

### Jupyter notebook installation
In order to run the jupyter notebook analyses in [overview.ipynb](overview.ipynb), users should also install [jupyter-lab](https://jupyter.org/install) and then ipykernel, in order to add your conda environment to jupyter-lab to use with the analyses.

    python -m pip install jupyterlab
    conda install ipykernel
    ipython kernel install --user --name=name_of_environment

then refresh notebook to see and change the jupyter notebook's kernel to your new conda environment in top right corner.

## Usage
Users are encouraged to view the [overview.ipynb](overview.ipynb) notebook, which shows the import and use of the `imgtofmri.predict()` function, as well as its extension to movies using `utils.extract_frames()` and `utils.load_frames()`.

**imgtofmri** can also be used as a command-line interface, as:

        python imgtofmri.py [-h] --input input_dir [--output output_dir] [--rois each roi here]
                            [--sigma sigma_val] [--center_crop center_crop]

## Support or Questions
Users are encouraged to email [Max Bennett](mailto:mbb2176@columbia.edu) with questions or issues.

## License
This package is licensed under an MIT license in `LICENSE.txt`