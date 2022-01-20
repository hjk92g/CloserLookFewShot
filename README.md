# Distance-Ratio-Based Formulation for Metric Learning

## Environment
 - Python3
 - Pytorch (http://pytorch.org/) (version 1.6.0+cu101)
 - json
 - tqdm

## Preparing datasets
### CUB
* Change directory to `/filelists/CUB`
* run `source ./download_CUB.sh`

One might need to manually download CUB data from http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz.

### mini-ImageNet
* Change directory to `/filelists/miniImagenet`
* run `source ./download_miniImagenet.sh` (WARNING: This would download the 155G ImageNet dataset.) 

To only download 'miniImageNet dataset' and not the whole 155G ImageNet dataset:

(Download 'csv' files from the codes in `/filelists/miniImagenet/download_miniImagenet.sh`. Then, do the following.)

First, download zip file from https://drive.google.com/file/d/0B3Irx3uQNoBMQ1FlNXJsZUdYWEE/view (It is from https://github.com/oscarknagg/few-shot). After unzipping the zip file at `/filelists/miniImagenet`, run a script ```/filelists/miniImagenet/prepare_mini_imagenet.py``` which is modified from https://github.com/oscarknagg/few-shot/blob/master/scripts/prepare_mini_imagenet.py. Then, run ```/filelists/miniImagenet/write_miniImagenet_filelist2.py```.

## Train
Run
```python ./train.py --dataset [DATASETNAME] --model [BACKBONENAME] --method [METHODNAME] --train_aug [--OPTIONARG]```

To also save training analyses results, for example, run `python ./train.py --dataset miniImagenet --model Conv4 --method protonet_S --train_aug --n_shot 5 --train_n_way 5 --test_n_way 5 > record/miniImagenet_Conv4_proto_S_5s5w.txt`  

```train_models.ipynb``` contains codes for our experiments.

## Save features
Save the extracted feature before the classifaction layer to increase test speed. 

For instance, run
```python ./save_features.py --dataset miniImagenet --model Conv4 --method protonet_S --train_aug --n_shot 5 --train_n_way 5```

## Test
For example, run
```python ./test.py --dataset miniImagenet --model Conv4 --method protonet_S --train_aug --n_shot 5 --train_n_way 5 --test_n_way 5```

## Analyze training
Run ```/record/analyze_training_1shot.ipynb``` and ```/record/analyze_training_5shot.ipynb``` to analyze training results (norm ratio, con-alpha ratio, div-alpha ratio, and con-div ratio)

## Results
The test results will be recorded in `./record/results.txt`

## Visual comparison of softmax-based and distance-ratio-based (DR) formulation
The following images visualize confidence scores of red class when the three points are the representing points of red, green, and blue classes.
 | Softmax-based formulation | DR formulation |
 | :---:       |     :---:      |
 | <img src="https://github.com/hjk92g/DR_Formulation_ML/blob/master/plots/prob_red_softmax_sq.png" width="360" height="300" /> | <img src="https://github.com/hjk92g/DR_Formulation_ML/blob/master/plots/prob_red_DR_p_2.png" width="360" height="300" /> | 

## References and licence
Our repository (a set of codes) is forked from an original repository (https://github.com/wyharveychen/CloserLookFewShot) and codes are under the same licence (```LICENSE.txt```) as the original repository except for the following.

```/filelists/miniImagenet/prepare_mini_imagenet.py``` file is modifed from https://github.com/oscarknagg/few-shot. It is under a different licence in ```/filelists/miniImagenet/prepare_mini_imagenet.LICENSE```

Copyright and licence notes (including the copyright note in ```/data/additional_transforms.py```) are from the original repositories (https://github.com/wyharveychen/CloserLookFewShot and https://github.com/oscarknagg/few-shot). 

## Modifications
List of modified or added files (or folders) compared to the original repository (https://github.com/wyharveychen/CloserLookFewShot):

```io_utils.py```
```backbone.py```
```configs.py```
```train.py```
```save_features.py```
```test.py```
```utils.py```
```README.md```
```train_models.ipynb```
```/methods/__init__.py```
```/methods/protonet_S.py```
```/methods/meta_template.py```
```/methods/protonet_DR.py```
```/methods/softmax_1nn.py```
```/methods/DR_1nn.py```
```/models/```
```/filelists/miniImagenet/prepare_mini_imagenet.py```
```/filelists/miniImagenet/prepare_mini_imagenet.LICENSE```
```/filelists/miniImagenet/write_miniImagenet_filelist2.py```
```/record/```
```/record/preprocessed/```
```/record/analyze_training_1shot.ipynb```
```/record/analyze_training_5shot.ipynb```

My (Hyeongji Kim) main contributions (modifications) are in ```/methods/meta_template.py```, ```/methods/protonet_DR.py```, ```/methods/softmax_1nn.py```, ```/methods/DR_1nn.py```, ```/record/analyze_training_1shot.ipynb```, and ```/record/analyze_training_5shot.ipynb```.
