# img2fmri

**img2fmri**: a python package for predicting group-level fMRI responses to visual stimuli using deep neural networks

[![PyPI version](https://badge.fury.io/py/img2fmri.svg)](https://badge.fury.io/py/img2fmri) [![Documentation Status](https://readthedocs.org/projects/img2fmri/badge/?version=latest)](https://img2fmri.readthedocs.io/en/latest/?badge=latest)

<img src="https://raw.githubusercontent.com/dpmlab/img2fmri/main/model_overview.png" width="700" class="center"/>

Our model is intended to be used by researchers that might benefit from analyzing stimulus-driven patterns of activity in visual cortex, including research involving the temporal dynamics of visual cortex, or research conducted in the absence of an fMRI scanner and human subjects.

Users are encouraged to read the background science information for an overview of the model and its intended uses ([science_overview.pdf](science_overview.pdf)).

## Installation
To install and use img2fmri, users must work from a coding environment with [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation) and [AFNI](https://afni.nimh.nih.gov/pub/dist/doc/htmldoc/background_install/main_toc.html) installed or use our Docker container which comes with those packages pre-installed. 

### pip/PyPI

For users who already have FSL and AFNI installed, img2fmri can be installed and tested with:

    pip install img2fmri
    pytest -s --pyargs img2fmri

### Docker

For users that intend to use our docker environment to test and use img2fmri, the Dockerfile included in this repo can be used to build an image as follows:

    docker build --no-cache --tag img2fmri --file Dockerfile .

and if successfully built, can be run and tested with:

    docker run -it -p 8888:8888 img2fmri
    pytest -s --pyargs img2fmri

Alternatively, our pre-built image can be pulled and used,though do note that this is a large (~10GB compressed) image.

    docker pull mbennett12/img2fmri
    docker run -it -p 8888:8888 mbennett12/img2fmri
    pytest -s --pyargs img2fmri

### Conda

Optionally, users can install [Conda](https://docs.conda.io/en/latest/) and create an environment with python 3.9 ready for img2fmri (NOTE: this conda environment still needs access to FSL and AFNI): 

    conda create --name name_of_environment python=3.9
    conda activate name_of_environment
    pip install img2fmri

### (Optional) Jupyter and running analyses
In order to run the jupyter notebook analyses in [overview.ipynb](overview.ipynb) or 
[model_training.ipynb](model_training/model_training.ipynb), the following commands should be run.

If running the docker container from the command:

    docker run -it -p 8888:8888 img2fmri

users should then clone this repo and run the following command from within the container:

    git clone https://github.com/dpmlab/img2fmri.git
    python3 -m notebook --allow-root --no-browser --ip=0.0.0.0

in order to then access their docker container's jupyter notebook at the following url: `http://localhost:8888`. 
Note that users will need to copy and paste the token shown in the output of the previous command in their
 web browser to access their docker container's directory.

## Usage and Documentation
Users are encouraged to view our [ReadTheDocs documentation](https://img2fmri.readthedocs.io/en/latest/) 
for our API documentation, and also review [overview.ipynb](overview.ipynb) notebook which shows the import 
and use of the `img2fmri.predict()` function, as well as its extension to movies using 
`img2fmri.predict(predict_movie=True)`.

**img2fmri** can also be used as a command-line interface, as:

    img2fmri [-h] --input input_dir_or_movie [--output output_dir]
             [--roi_list each roi here] [--sigma sigma_val] [--center_crop true_or_false]
             [--predict_movie true_or_false]
             
### Regions of Interests (ROIs) and neuroimaging files
fMRI ROI bool masks (in subject space, MNI space, and MNI space post-smoothing), T1 files, and other 
neuroimaging reference files that are used in our prediction pipeline can be found in the [derivatives](derivatives) folder.
             
## Testing
Users can test their img2fmri installation by using the following command:

    pytest -s --pyargs img2fmri

which will run the tests located in `img2fmri/tests/run_test.py`, which test the python-imported image prediction, the movie prediction pipeline, and the command line interface package usage.

## Support and Contributing
Users are encouraged to review [CONTRIBUTING.rst](CONTRIBUTING.rst) with suggestions on how to report issues and contribute to the img2fmri software package. Users can also email the author at [Max Bennett](mailto:mbb2176@columbia.edu) with questions or issues.

## License
This package is licensed under an MIT license found in [LICENSE.txt](LICENSE.txt)