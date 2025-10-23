# Dinomaly

PyTorch Implementation of
"Dinomaly: The Less Is More Philosophy in Multi-Class Unsupervised Anomaly Detection".
[paper](https://arxiv.org/abs/2405.14325)

The code is preview version, so it could be really ugly with minor errors. We will try to reformat it after the paper is accepted.

Give me a star if you like it!!!

News 09.26.2024: Just got rejected even with pretty positive scores. Wish me good luck.

## 1. Environments

Create a new conda environment and install required packages.

```
conda create -n my_env python=3.8.12
conda activate my_env
pip install -r requirements.txt
```
Experiments are conducted on NVIDIA GeForce RTX 3090 (24GB). Same GPU and package version are recommended. 

## 2. Prepare Datasets
Noted that `../` is the upper directory of Dinomaly code. It is where we keep all the datasets by default.
You can also alter it according to your need, just remember to modify the `data_path` in the code. 

### MVTec AD

Download the MVTec-AD dataset from [URL](https://www.mvtec.com/company/research/datasets/mvtec-ad).
Unzip the file to `../mvtec_anomaly_detection`.
```
|-- mvtec_anomaly_detection
    |-- bottle
    |-- cable
    |-- capsule
    |-- ....
```


### VisA

Download the VisA dataset from [URL](https://github.com/amazon-science/spot-diff).
Unzip the file to `../VisA/`. Preprocess the dataset to `../VisA_pytorch/` in 1-class mode by their official splitting 
[code](https://github.com/amazon-science/spot-diff).

You can also run the following command for preprocess, which is the same to their official code.

```
python ./prepare_data/prepare_visa.py --split-type 1cls --data-folder ../VisA --save-folder ../VisA_pytorch --split-file ./prepare_data/split_csv/1cls.csv
```
`../VisA_pytorch` will be like:
```
|-- VisA_pytorch
    |-- 1cls
        |-- candle
            |-- ground_truth
            |-- test
                    |-- good
                    |-- bad
            |-- train
                    |-- good
        |-- capsules
        |-- ....
```
 
### Real-IAD
Contact the authors of Real-IAD [URL](https://realiad4ad.github.io/Real-IAD/) to get the net disk link.

Download and unzip `realiad_1024` and `realiad_jsons` in `../Real-IAD`.
`../Real-IAD` will be like:
```
|-- Real-IAD
    |-- realiad_1024
        |-- audiokack
        |-- bottle_cap
        |-- ....
    |-- realiad_jsons
        |-- realiad_jsons
        |-- realiad_jsons_sv
        |-- realiad_jsons_fuiad_0.0
        |-- ....
```

## 3. Run Experiments
Multi-Class Setting
```
python dinomaly_mvtec_uni.py --data_path ../mvtec_anomaly_detection
```
```
python dinomaly_visa_uni.py --data_path ../VisA_pytorch/1cls
```
```
python dinomaly_realiad_uni.py --data_path ../Real-IAD
```

Conventional Class-Separted Setting
```
python dinomaly_mvtec_sep.py --data_path ../mvtec_anomaly_detection
```
```
python dinomaly_visa_sep.py --data_path ../VisA_pytorch/1cls
```
```
python dinomaly_realiad_sep.py --data_path ../Real-IAD
```

Training Unstability: The optimization can be unstable with loss spikes (e.g. ...0.05, 0.04, 0.04, **0.32**, **0.23**, 0.08...)
, which can be harmful to performance. This occurs very very rare. If you see such loss spikes during training, consider change a random seed.
