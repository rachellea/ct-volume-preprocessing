# ct-volume-preprocessing

ct-volume-preprocessing includes:
* preprocess_volumes.py, an end-to-end Python
pipeline for complete preprocessing of computed tomography (CT) scans from
DICOM format to clean numpy arrays suitable for machine learning.
* download_volumes.py, an example Python pipeline for downloading CT data in bulk.
* visualize_volumes.py, which contains Python code for visualizing CT scans in
an interactive way, visualizing CT scans as GIFs, and making other figures
from CT data.

## Requirements

All pipelines are implemented in Python and can be run using the Singularity
container or the requirements defined here:
https://github.com/rachellea/research-container

I've also included a requirements.txt in this repo which was
obtained by pruning out everything I don't think is necessary from the
complete requirements.txt provided in the research-container repo. 

## Details: Preprocessing CT Data

The steps of the CT volume preprocessing pipeline are described in detail in
"Appendix A.2 CT Volume Preparation" of our [Medical Image Analysis paper](https://doi.org/10.1016/j.media.2020.101857).
The paper is also available [on arXiv](https://arxiv.org/ftp/arxiv/papers/2002/2002.04752.pdf).

If you find this work useful in your research, please consider citing us:

Draelos, Rachel Lea, et al. "Machine-Learning-Based Multiple Abnormality Prediction with Large-Scale Chest Computed Tomography Volumes." *Medical Image Analysis* (2020).

The CleanCTScans class in preprocess_volumes.py assumes that the CT scans
to be processed are saved in one directory and that each CT scan is saved as
a pickled Python list. Each element of the list is a pydicom.dataset.FileDataset
that represents a DICOM file and thus contains metadata as well as pixel data.
Each pydicom.dataset.FileDataset corresponds to one slice of the CT scan.
The slices are not necessarily 'in order' in this list.

Note that if your CT scans are instead stored as raw DICOMs with one DICOM per
slice, you can easily modify the pipeline to first read each DICOM file into a
pydicom.dataset.FileDataset directly using pydicom. Then you can aggregate these
into a list to use the pipeline on your data.

For each CT scan, preprocess_volumes.py will order the slices and stack
them into a volume, rescale pixel values to Hounsfield Units (HU), clip
the pixel values to [-1000 HU, +1000 HU], resample to spacing of
0.8 x 0.8 x 0.8 mm, and save the final CT volume as a zip-compressed
numpy array of 16-bit integers. These numpy arrays can then be loaded
as input data for machine learning with PyTorch or TensorFlow.

(Note that ML-specific preprocessing like normalizing pixel values is
not part of this particular pipeline, as those steps are quick and often are
customized to a particular data set.)

## Details: Downloading CT Data in Bulk

download_volumes.py is included as an example of a pipeline for downloading
CT scans in bulk. This pipeline will not run because the required endpoints,
IDs, and tokens have all been removed for security reasons. The code is only
included as an example.

A CT scan is associated with multiple series. This download pipeline includes
logic for selecting the original series with the greatest number of slices.

## Credits

The slice ordering step uses code modified from Innolitics' dicom-numpy repo:

https://github.com/innolitics/dicom-numpy/blob/master/dicom_numpy/combine_slices.py

This dicom-numpy code was originally downloaded on September 19, 2019.
