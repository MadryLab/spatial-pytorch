# Exploring the Landscape of Spatial Robustness

This repository is the codebase for the ICML 2019 paper [Exploring the Landscape
of Spatial
Robustness](http://proceedings.mlr.press/v97/engstrom19a/engstrom19a.pdf). 

## Overview

Our code is based heavily on the [Robustness framework](https://github.com/madrylab/robustness). For documentation refer to this library's repository. The presented code differs in a few key ways from the original code used to calculate the results in the paper. 

- It was written in PyTorch rather than TensorFlow
- We only support ImageNet rather than ImageNet, CIFAR, and MNIST
- It uses a ResNet18 rather than a ResNet50 (due to time constraints; ResNet50 are forthcoming)
- The data augmentation is slightly different. The original code used data augmentation from [here](https://github.com/tensorpack/tensorpack/blob/f20e411a7cb28c4b2004bd66c4392d45071e0096/examples/ResNet/imagenet_utils.py#L53). This data augmentation is nearly identical to that used in this repository, except that the resize/cropping algorithms are slightly different, and the lighting applied has slightly different parameters. The data augmentation routine used can be found in detail in the repository [here](https://github.com/MadryLab/spatial-pytorch/blob/0c9b3c608047989d72d6fed7ec3235c360e88857/robustness/data_augmentation.py#L34).
- For LR: We use 110 epochs with a learning rate schedule of 0.1 from 0 to 29, and then order of magnitude drops in learning rate at (inclusively) steps 30, 60, 85, 95, and 105. Ultimately the learning rate reduces to 1e-6. 
- The original code used a `tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)` as the optimizer. This release uses SGD with standard (not nesterov) momentum.

## Model weights

Expand the zip file [here](https://www.dropbox.com/s/briburs7dl0qslf/out.zip?dl=0) to get the model weights.

## Training

Training is completed using `./train.sh`. Before training or evaluating you need
to set the environmental variable `$DATA` to be the path to the ImageNet data
directory (where the format is the [typical PyTorch ImageNet
format](https://github.com/pytorch/examples/tree/master/imagenet#requirements)).
There are three arguments:

- You can set the training mode (refer to paper for details) using `nocrop` (no random cropping), `standard` (standard training), `random` (just rotation/translations added to data augmentation), or `worst10` (choose worst of 10 transformations of a given image to train on) as the first argument.
- You can train using either 30 degree rotations / 24px translations or 40 degree rotations / 32px translations by using making 30 or 40 (respectively) the second argument.
- The third argument corresponds to the out directory

For example one training call could be:

    ./train.sh worst10 30 models/

## Evaluation
Before evaluating you need to set the environmental variable `$DATA` to be the
path to the ImageNet data directory (where the format is the [typical PyTorch
ImageNet
format](https://github.com/pytorch/examples/tree/master/imagenet#requirements)).
The command for evaluating a model is:

    ./eval.sh $TRAINING_TYPE $CONSTRAINT $ATTACK_TYPE $MODELS_DIR $EVAL_ATTACK

Where:

- `$TRAINING_TYPE` is as above
- `$CONSTRAINT` is as above (these first two arguments are just to figure out which model to grab for evaluation)
- `$ATTACK_TYPE` is the type of attack to use in the evaluation of the model
- `$MODELS_DIR` is the directory full of model folders to use 

## Benchmarks
Below is a list of various models (rows) and their computed accuracies under
each attack (columns). The trends and accuracies are slightly different from the
ones found in the original paper, see above for the key differences.

|                           | Natural | Random (30 deg/24px) | Worst-of-10 Random | Exhaustive Search |
|---------------------------|:-------:|:--------------------:|:------------------:|:-----------------:|
| Standardly Trained Model  |  71.3%  |        60.6%         |       39.9%        |       20.9%       |
|          No Crop          |  65.4%  |        54.1%         |       30.5%        |       11.8%       |
|  Data Aug. (30 deg/24px)  |  68.9%  |        68.0%         |       54.6%        |       38.1%       |
|  Data Aug. (40 deg/32px)  |  68.3%  |        67.7%         |       54.9%        |       38.8%       |
| Worst-of-10 (30 deg/24px) |  69.1%  |        68.0%         |       59.4%        |       48.1%       |
| Worst-of-10 (40 deg/32px) |  68.6%  |        67.2%         |       59.3%        |       48.4%       |

## Citation
When citing this work, you should use the following bibtex:

    @inproceedings{engstrom2019exploring,
      title={Exploring the Landscape of Spatial Robustness},
      author={Engstrom, Logan and Tran, Brandon and Tsipras, Dimitris and Schmidt, Ludwig and Madry, Aleksander},
      booktitle={International Conference on Machine Learning},
      pages={1802--1811},
      year={2019}
    }


