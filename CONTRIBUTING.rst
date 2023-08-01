Goals
=====

We've built a python package and command line interface (CLI) to predict 
group-averaged fMRI responses to visual stimuli (images and movies).
We encourage other researchers to expand the prediction pipeline to other 
feature extraction/machine learning models, other regions of visual cortex,
and even new fMRI training datasets. Our intent is that our package can be
expanded upon and reused for varied neuroscience use cases.

Support and Issues
==================

Users are encouraged to first view our API documentation linked from our README,
and also encouraged to view ``overview.ipynb`` and ``analysis.py`` to see example 
uses of ``img2fmri``. 

Further issues and support can be handled on a case-by-case basis, and users are 
encouraged to create a "New Issue" in the Issue tab on our Github repository, or
to email the author at mbb2176@columbia.edu for questions.

How to contribute
=================

We use GitHub pull requests (PRs) to make improvements to the repository. You
should make a fork, create a new branch for each new feature you develop, and
make a PR to merge your branch into the main branch of the official
repository. There are several workflows you could follow. Here is a concise
step-by-step description of our recommended workflow:

1. Fork the official img2fmri repository on GitHub.

2. Clone your fork::

     git clone https://github.com/yourgithubusername/img2fmri

3. Add the official img2fmri repository as the ``upstream`` remote::

     git remote add upstream https://github.com/dpmlab/img2fmri

4. Set the ``master`` branch to track the ``upstream`` remote::

     git fetch upstream
     git branch -u upstream/master

5. Whenever there are commits in the official repository, pull them to keep
   your ``master`` branch up to date::

     git pull --ff-only

6. Always create a new branch when you start working on a new feature; we only
   update the ``master`` branch via pull requests from feature branches; never
   commit directly to the ``master`` branch::

     git checkout -b new-feature

7. Make changes and commit them. Include a news fragment for the release notes
   in ``docs/newsfragments`` if your changes are visible to users (see `Pip's
   documentation`_ and our news types in ``pyproject.toml``).

8. Push your feature branch to your fork::

     git push --set-upstream origin new-feature  # only for the first push
     git push  # for all subsequent pushes

9. When your feature is ready, make a PR on GitHub. If you collaborate with
   others on the code, credit them using Co-authored-by_; if you are merging a
   PR, credit all authors using Co-authored-by_ in the PR squash message. After
   your PR is merged, update your ``master`` branch and delete your feature
   branch::

     git checkout master
     git pull --ff-only
     git branch -d new-feature
     git push --delete origin new-feature  # or use delete button in GitHub PR

Please see the `GitHub help for collaborating on projects using issues and pull
requests`_ for more information.

.. _Pip's documentation:
   https://pip.pypa.io/en/latest/development/#adding-a-news-entry
.. _GitHub help for collaborating on projects using issues and pull requests:
   https://help.github.com/categories/collaborating-on-projects-using-issues-and-pull-requests/
.. _Co-authored-by:
   https://help.github.com/en/github/committing-changes-to-your-project/creating-a-commit-with-multiple-authors

You should test your contributions on your computer by first locally installing your 
module using ``pip install -e .`` from within your working directory, and then using
``pytest -s --pyargs img2fmri`` before submitting a PR.

If you want to obtain early feedback for your work, ask people to look at your
fork. Alternatively, you can open a PR before your work is ready; in this case,
you should start the PR title with ``WIP:``, to let people know your PR is work
in progress.

Model and ROI Extension
=======================
We encourage developers to extend our package to other feature extraction models and to other 
visual cortex ROIs, and include a comment on where to change the feature extraction model in the 
predict.generate_activations() function. This can also be found in the 
model_training/model_training.ipynb notebook.

If extending to new ROIs and new training participants, we note that developers will need to adapt 
our pipeline from predicting responses in three subject's brain spaces into n number of subject's spaces,
and furthermore, adapt the transformation to MNI pipeline accordingly. Lastly, we note that the 
utils.get_subj_overlap() function should be updated to include these additional ROIs, and the union 
of their bool masks accordingly. Developers interested in extending the model in this manner are 
encouraged to reach out to the authors for collaboration.

Standards
=========

* Python docstrings should be formatted according to the NumPy docstring
  standard as implemented by the `Sphinx Napoleon extension`_ (see also the
  `Sphinx NumPy example`_). In particular, note that type annotations must
  follow `PEP 484`_. Please also read the `NumPy documentation guide`_, but
  note that we consider Sphinx authoritative.

.. _Sphinx Napoleon extension:
   http://www.sphinx-doc.org/en/stable/ext/napoleon.html
.. _Sphinx NumPy example:
   http://www.sphinx-doc.org/en/stable/ext/example_numpy.html
.. _NumPy documentation guide:
   https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt

* All code exposed through public APIs must have documentation that explains
  what the code does, what its parameters mean, and what its return values can
  be, at a minimum.


Acknowledgements
================

Thanks to the BRAINIAK_ team for their CONTRIBUTING.rst document, which we
base ours on.
.. _BRAINIAK: https://github.com/brainiak/brainiak/tree/master