# Computer Pointer Controller

| Details of Software |                    |                   
|---------------------|--------------------|
| OS:                 | Ubuntu\* 18.04 LTS |
| OpenVINO:           | 2020.3 LTS         |
| Python:             |  3.6.9             |

## Introduction

The app of Computer Pointer Controller is applied to control the movement of mouse pointer by using gaze detection points.

It is a demonstration to perform complete OpenVINO pipeline from leveraging a pre-trained model with Intel® open model zoo (conduct model optimization if necessary), programming for the preprocessing and post-processing steps to proceed inference, and finally deploy to the edge.

Specifically, we’ll be using 4 different models as following to run this application, which will be detailed addressed further.

- Face Detection Model
- Facial Landmarks Detection Model
- Head Pose Estimation Model
- Gaze Estimation Model

![workflow](./bin/pipeline.png)

## Project Set Up and Installation

### Prerequisites
Check out this [link](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/download.html#operatingsystem=Linux*&#distributions=Web%20Download&#options=Online) to get the Intel® Distribution of OpenVINO™ Toolkit. The detailed guide of installing it for Linux* OS can be referred to [here](https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_linux.html).

![openvino_logo](./bin/openvino.jpg)

Once the installation has been successfully done, we’re good to build this project.

### Build
#### Step 1: Clone this repository to your workspace
```
git clone https://github.com/jonathanyeh0723/OpenVINO_Computer_Pointer_Controller
```

#### Source the OpenVINO environment
```
source /opt/intel/openvino/bin/setupvars.sh
```
We should be able to see the following returned, if it is successful:
```
[setupvars.sh] OpenVINO environment initialized
```
