# SeesawNet-pytorch-reimplement
Pytorch(0.4.1/1.0 verified) codes and pre-trained models for the paper: Seesaw-Net: Convolution Neural Network With Uneven Group Convolution,  https://arxiv.org/abs/1905.03672v4.

Seesawnet, inspired by IGCV3, could achieve better performance on cifar10/cifar100/imagenet than mobilenetv2/IGCV3/shufflenetv2(but with more flops). Due to limited resources(AWS), we only trained imagenet-1k two times-Seesawnet(72.9%) and Seesawnet_0.5D. All pretrained model mentioned within the paper will be released.

ImageNet
The training module comes from https://github.com/pytorch/examples, that is our implement does not include extra tricks since we would perfer reproduce performance reported by papers should be easy.

CIFAR10/CIFAR100
The training module partly comes from this repo:https://github.com/xxradon/IGCV3-pytorch, but we correct the IGCV3 model and training related paras reffer to original paper and could replemenet the performance of IGCV3 as our baseline.

Main Dependencies
pytorch==0.3.1
numpy 
tqdm 
easydict
matplotlib 
tensorboardX 

We finished this work by July 2018 and before shufflenetv2 announced. however, due to limited leisure time(we achieve another pro version of SeesawNet and verified all related models on image segmentation(deeplabv3)/detection(ssd variants)/style transfer(replace backbone)/face recognition) and little experience on paper writing, we directlly post it on arxiv later.
