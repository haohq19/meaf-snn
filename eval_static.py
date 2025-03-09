# pretrain spiking neural networks on the ES-ImageNet dataset

import os
import argparse
import torch
import torch.nn as nn
import models.sew_resnet_static as sew_resnet
import models.spiking_resnet_static as spiking_resnet
from torch.utils.data import DataLoader
from utils.dist import init_dist
from torchvision.datasets import ImageNet
from torchvision import transforms
import tqdm
from torch.utils.tensorboard import SummaryWriter
from spikingjelly.activation_based import functional

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nsteps', default=4, type=int, help='number of time steps')
    parser.add_argument('--num_classes', default=1000, type=int, help='number of classes')
    parser.add_argument('--model', default='spiking_resnet34', type=str, help='model type')
    parser.add_argument('--connect_f', default='ADD', type=str, help='spike-element-wise connect function')
    parser.add_argument('--device_id', default=7, type=int, help='GPU id to use')
    parser.add_argument('--pt_dir', default='weights/static/', type=str, help='path to pretrained weights')
    return parser.parse_args()

def _get_model(args):
    # load pretrained weights
    pt_path = os.path.join(args.pt_dir, args.model + '.pth')
    if not os.path.exists(pt_path):
        raise FileNotFoundError(pt_path)
    else:
        checkpoint = torch.load(pt_path)
        if 'model' in checkpoint.keys():
            pt_weights = checkpoint['model']
        else:
            pt_weights = checkpoint

    # model
    if args.model in sew_resnet.__dict__:
        model = sew_resnet.__dict__[args.model](num_classes=1000, T=args.nsteps, connect_f=args.connect_f)
        if pt_weights is not None:
            state_dict = {k.replace('module.', ''): v for k, v in pt_weights.items()}
            # state_dict = {k.replace('fc', 'fc.0'): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            print('Load pretrained weights from [{}]'.format(pt_path))
    elif args.model in spiking_resnet.__dict__:
        model = spiking_resnet.__dict__[args.model](num_classes=1000, T=args.nsteps)
        if pt_weights is not None:
            state_dict = {k.replace('module.', ''): v for k, v in pt_weights.items()}
            # state_dict = {k.replace('fc', 'fc.0'): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            print('Load pretrained weights from [{}]'.format(pt_path))

    else:
        raise NotImplementedError(args.model)
    
    for param in model.parameters():
        param.requires_grad = False

    return model




def validate(
    model: nn.Module,
    criterion: nn.Module,
    data_loader: DataLoader,
):
    torch.cuda.empty_cache()
    model.eval()

    top1_correct = 0
    top5_correct = 0
    n_samples = 0
    total_loss = 0

    # progress bar
    process_bar = tqdm.tqdm(total=len(data_loader))

    with torch.no_grad():
        for step, (data, label) in enumerate(data_loader):
            input = data.cuda(non_blocking=True)
            target = label.cuda(non_blocking=True)

            output = model(input)
            loss = criterion(output, target)
            functional.reset_net(model)

            # calculate the top5 and top1 accurate numbers
            _, predicted = output.cpu().topk(5, 1, True, True)  # batch_size, topk(5) 
            top1_correct += predicted[:, 0].eq(label).sum().item()
            top5_correct += predicted.T.eq(label[None]).sum().item()
            total_loss += loss.item() * len(label)
            n_samples += len(label)
            process_bar.update(1)
    
    top1_acc = top1_correct / n_samples
    top5_acc = top5_correct / n_samples
    process_bar.close()

    print('val: acc@1: {:.5f}, acc@5: {:.5f}, loss: {:.6f}, cor@1: {}, cor@5: {}, total: {}'.format(    
         top1_acc, top5_acc, total_loss / n_samples, top1_correct, top5_correct, n_samples
        )
    )

    return top1_acc

def main(args):

    # init distributed training
    init_dist(args)
    print(args)
    
    # device
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
    else:
        torch.cuda.set_device(args.device_id)

     # data
    dataset = ImageNet(root='/home/haohq/datasets/ImageNet', split='val', transform=transform)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=8, pin_memory=True)

    # criterion
    criterion = nn.CrossEntropyLoss()
    args.criterion = criterion.__class__.__name__

    # model
    model = _get_model(args)
    
    model.cuda()

    val_acc = validate(
        model=model,
        criterion=criterion,
        data_loader=dataloader,
    )
    print('val_acc: {:.5f}'.format(val_acc))
    

if __name__ == '__main__':
    args = parser_args()
    main(args)