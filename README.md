# KinD
This is a Tensorflow implementation of KinD

Kindling the Darkness: a Practical Low-light Image Enhancer. In ACMMM2019<br>
Yonghua Zhang, Jiawan Zhang, Xiaojie Guo

### Requirements ###
1. Python
2. Tensorflow >= 1.10.0
3. numpy, PIL

### Test ###
First download the pre-trained checkpoints from [here](https://pan.baidu.com/s/1c4ZLYEIoR-8skNMiAVbl_A), then just run
```shell
python evaluate.py
```
### Train ###
Please download the [LOLdataset](https://daooshee.github.io/BMVC2018website/). Save training pairs of LOL dataset under './LOLdataset/our485/' and save evaluating pairs under './LOLdataset/eval15/'. For training, just run
```shell
python decomposition_net_train.py
python adjustment_net_train.py
python reflectance_resoration_net_train.py
```
You can also evaluate on the LOLdataset, just run
```shell
python evaluate_LOLdataset.py
```
Our code partly refers to the [code](https://github.com/weichen582/RetinexNet).

### Citation ###
```
@inproceedings{zhang2019kindling,
  title="Kindling the Darkness: A Practical Low-light Image Enhancer.",
  author="Yonghua {Zhang} and Jiawan {Zhang} and Xiaojie {Guo}",
  booktitle="Proceedings of the 27th ACM International Conference on Multimedia  - MM '19",
  pages="1632--1640",
  year="2019"
}
```
