# Source-Free Active Domain Adaptation for Medical Image Segmentation via Influential Points Learning

This repository contains the supported pytorch code and configuration files to reproduce of BDK.

![BDK](imgs/Methods.jpg?raw=true)

# Introduction
This project propose a novel SFADA method for medical image segmentation.  

We constructed neighborhoods based on semantic similarity and generated an informative score ranking for all target samples. The top target samples in this ranking were manually annotated. These annotated and remaining unlabeled samples were then fed into a semi-supervised learning pipeline with two adaptation stages, which enabled a progressively stable DA process.  

Our method was validated using a multi-center nasopharyngeal carcinoma segmentation dataset and a prostate segmentation dataset. Experimental results showed that our method achieved comparable accuracy to the fully supervised upper bound, even with only 20% of the annotation. Meanwhile, compared to state-of-the-art methods, our approach exhibited marked advancements under lower annotation budgets.

## Environment

Please prepare an environment with Python 3.7, Pytorch 1.7.1.  
`conda create --name SFADA --file requirements.txt`

## Dataset Preparation

Datasets can be acquired via following links:

- NPC Dataset: [Source domain](https://zenodo.org/records/10900202), [target domain](https://pan.baidu.com/share/init?surl=lIRmboirlEPm2HrKe5SyQQ).
- Prostate Dataset: [Source and target domain](https://liuquande.github.io/SAML/).


## Preprocess Data
Convert nii.gz Files to h5 Format to facilitate follow-up processing and training  
`python dataloaders/data_processing.py`


# Usage
### 1. Training source models in a single center  
`python train_single_center.py`

### 2. Select Active Samples Using BDK  
`python bdk_select.py`

### 3. First Fine-tune the source model for course adaptation  
`python train_multi_center_finetune.py`

### 4. Generate pseudo labels for the unlabeled samples  
`python pseudo_generate.py`

### 5. Second Fine-tune the source model for fine adaptation  
`python train_multi_center_finetune.py`

### 6. Testing on the target domain  
`python test.py`


# Results

![BDK](imgs/Segmentation.jpg?raw=true)


## Acknowledge
Parts of codes are borrowed from [STDR](https://github.com/whq-xxh/SFADA-GTV-Seg).

