# KinD
This is a Tensorflow implement of KinD

Kindling the Darkness: a Practical Low-light Image Enhancer. In ACMMM2019<br>
Yonghua Zhang, Jiawan Zhang, Xiaojie Guo

### Requirements ###
1. Python
2. Tensorflow >= 1.10.0
3. numpy, PIL

### Test ###
First download the pre-trained checkpoints from [here](https://pan.baidu.com/s/1c4ZLYEIoR-8skNMiAVbl_A), then just run
```shell
python evalate.py
```
### Train ###
Please download the [LOLdataset](https://daooshee.github.io/BMVC2018website/). Save training pairs of LOL dataset under './LOLdataset/our485/' and evaling pairs under './LOLdataset/eval15/'. First training the decompositon net, then training the illumination adjustment net, finally training the restoration net. For example, just run
```shell
python evalate.py
```
