# WSI with Gaze Processing

Author: JIANG Maiqi

## Introduction
The whole slide image (WSI) is a kind of medical image with huge size. It is hard to process the whole slide image as general image. 
In this repository, we preprocess the data by splitting WSI to patches, following the doctor's gaze. The gaze is the doctor's attention to the WSI.
So, we can use the gaze to split the WSI to patches which are the input of deep learning model. 
The model can be used to predict the doctor's attention to the WSI and classify the WSI.

This repository is only focus on the preprocessing of WSI. The model is not included in this repository.

## Data Preparation

It is based on private wsi data, so we can not provide the data. There is another private data need to be prepared. The data is the 
doctor's gaze data. If you want to use your own gaze data, the gaze data loading can reference `./tool/dataProcessor.py` and `/tool/recReader.py`. 
These two files (provided by QIAO Siyu, Northeastern University, China) are used to load the gaze data. The gaze data is `.rec` file and `.erc` file. These files are binary 
files. They are file formats. 

## Environment

As for the installation, you need to check out the environment first. The environment is as follows:

```angular2html
python ~= 3.8.13
CUDA ~= 11.3
CuDNN ~= 8.2.0
(Optional) MiniConda
```

### Install OpenSlide

If you are using Windows, you can install OpenSlide by following the guidance from
[OpenSlide Website](https://openslide.org/api/python/#installing).

For linux, the installation is a little easier. For example, in ubuntu, you can just use the following command:

```angular2html
apt install python-openslide    
pip install Openslide-python
```
OpenSlide is a useful tool for reading WSI. 

### Install OpenCV

In order to process the WSI, you need to install OpenCV. In ubuntu, you can install it by following the command:

```angular2html
sudo apt-get install python3-opencv
```

In Windows, you can install it by following the command:

```angular2html
pip install opencv-python
pip install opencv-contrib-python
```

### Install Pytorch

Following your CUDA version, you can install Pytorch as suggestion command from 
[Pytorch Official Page](https://pytorch.org/get-started/locally/). For example, if you are using CUDA 11.3, you can install Pytorch by following the command:

```angular2html
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch  # if you are using conda
```

## Usage

You can import this repository to your project by following the command:

```
import sys
import_path = {repository folder}
if import_path[-1] == '/':
    import_path = import_path[:-1]
sys.path.append(import_path)
from StartProcess import FullProcess
```

The `FullProcess` is the main class of this repository. You can use it to preprocess the WSI. The usage is as follows:

```
process_func = FullProcess({gaze folder}, init_status=False)
base_dataset = process_func.dataset
```


