To generate the DAVIS-7 and UCF101-7 datasets used in VIDM paper evaluation,
please follow the following steps.

Both of these datasets are based on publicly available versions: DAVIS 2017 [1]
and UCF101 [2]. The script here downloads the datasets and preprocesses it for
VIDM problem setup (predicting 7 novel frames from a pair of input images).

Prerequisities
==============

To run the dataset generation script, first install the required packages:

sudo apt-get install pip
pip install absl-py
pip install mediapy
pip install opencv-python
pip install tensorflow
pip install tensorflow-datasets
pip install tfds-nightly

Running the script
==================

To generate both DAVIS-7 and UCF-101-7, run shell script:
create_dataset.sh

Depending your internet speed this will take some time (perhaps ~ 1h)
as it will download the data and resize and process the images. The most time
consuming part is the UCF101 data download (even though we use a small portion
of the dataset only).

Please monitor for any errors, to ensure that the datasets are complete.


[1] Jordi Pont-Tuset, Federico Perazzi, Sergi Caelles, Pablo Arbelaez,
    Alex Sorkine-Hornung, and Luc Van Gool. The 2017 davis challenge on video
    segmentation.
[2] Khurram Soomro, Amir Roshan Zamir, and Mubarak Shah. Ucf101: A dataset of
    101 human actions classes from videos in the wild. 2012
