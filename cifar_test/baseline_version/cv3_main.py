import torch.backends.cudnn as cudnn
import torch
from cifar10data import CIFAR10Data
from cifar100data import CIFAR100Data
#from MobileNetV2 import MobileNetV2
from IGCV3 import IGCV3
from train import Train
from utils import parse_args, create_experiment_dirs
from tensorboardX import SummaryWriter
#from data_parallel import DataParallel
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def main():
    # Parse the JSON arguments
    config_args = parse_args()

    # Create the experiment directories
    _, config_args.summary_dir, config_args.checkpoint_dir = create_experiment_dirs(
        config_args.experiment_dir)

    #model = MobileNetV2(config_args)
    model = IGCV3(config_args)
    #model = torch.nn.DataParallel(IGCV3(config_args))
    #dummy_input = torch.rand(1, 1, 224, 224) #假设输入13张1*28*28的图片
    #with SummaryWriter(comment='IGCV3') as w:
    #    w.add_graph(model, (dummy_input, ))
    print(model)
    if config_args.cuda:
        model = torch.nn.DataParallel(model).cuda()
        #model.cuda()
        #model = torch.nn.DataParallel(model)
        #model.cuda()
        #model = torch.nn.parallel.DistributedDataParallel(model)
        cudnn.enabled = True
        cudnn.benchmark = True

    #exit()
    print("Loading Data...")
    #data = CIFAR10Data(config_args)

    data = CIFAR100Data(config_args)
    print("Data loaded successfully\n")
    
    trainer = Train(model, data.trainloader, data.testloader, config_args)

    if config_args.to_train:
        try:
            print("Training...")
            trainer.train()
            print("Training Finished\n")
        except KeyboardInterrupt:
            pass

    if config_args.to_test:
        print("Testing...")
        trainer.test(data.testloader)
        print("Testing Finished\n")


if __name__ == "__main__":
    main()
