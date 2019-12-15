# SeesawNet-pytorch-reimplement

Pytorch(0.4.1/1.0 verified) codes and pre-trained models for the paper: Seesaw-Net: Convolution Neural Network With Uneven Group Convolution,  https://arxiv.org/abs/1905.03672v4.

Seesawnet, inspired by IGCV3, could achieve better performance on cifar10/cifar100/imagenet than mobilenetv2/IGCV3/shufflenetv2(but with more flops). Due to limited resources(AWS), we only trained imagenet-1k two times-Seesawnet(72.9%) and Seesawnet_0.5D. All pretrained model mentioned within the paper will be released.

ImageNet

The training module comes from https://github.com/pytorch/examples, that is our implement does not include extra tricks since we would perfer reproducing performance reported by papers should be easy.

![Image text](https://github.com/cvtower/SeesawNet_pytorch/tree/master/figures/imagenet_test.jpg)
![Image text](https://github.com/cvtower/SeesawNet_pytorch/tree/master/figures/efficiency_bench.jpg)

CIFAR10/CIFAR100

The training module partly comes from this repo:https://github.com/xxradon/IGCV3-pytorch(thanks very much to @xxradon), but we correct the IGCV3 model and training related paras refer to original paper and could reproduce the performance of IGCV3 as our baseline.
![image](https://github.com/cvtower/SeesawNet_pytorch/tree/master/figures/cifar10_test.jpg)
![image](https://github.com/cvtower/SeesawNet_pytorch/tree/master/figures/cifar100_test.jpg)

Main Dependencies

pytorch==0.3.1
numpy 
tqdm 
easydict
matplotlib 
tensorboardX 

We finished this work by July 2018 before shufflenetv2 announced. however, due to limited leisure time(we achieve another pro version of SeesawNet and verified all related models on image segmentation(deeplabv3)/detection(ssd variants)/style transfer(replace backbone)/face recognition tasks) and little experience on paper writing, we directlly post it on arxiv later. Based on seesawnet, we further improve the basic block refer to certain papers and design a face verification model named SeesawFaceNets for mobile platform, which achieve new SOTA record on several public datasets, please refer to (SeesawFaceNets: sparse and robust face verification model for mobile platform, https://arxiv.org/abs/1908.09124). And we will release SeesawFaceNets pytorch implement repo(based on this repo:https://github.com/TreB1eN/InsightFace_Pytorch, many thanks to @TreB1eN) later.

Citation

Please cite our papers in your publications if it helps your research:

@misc{zhang2019seesawnet,
    title={Seesaw-Net: Convolution Neural Network With Uneven Group Convolution},
    author={Jintao Zhang},
    year={2019},
    eprint={1905.03672},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}

@misc{zhang2019seesawfacenets,
    title={SeesawFaceNets: sparse and robust face verification model for mobile platform},
    author={Jintao Zhang},
    year={2019},
    eprint={1908.09124},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
