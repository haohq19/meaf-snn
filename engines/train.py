# engines for training spiking neural networks

import os
import tqdm
import argparse
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from spikingjelly.activation_based import functional
from utils.dist import is_master, save_on_master, global_meters_sum

# set random seed to be incosistent with the results of SEW_ResNet
_seed_ = 2020
random.seed(2020)
torch.manual_seed(_seed_)
torch.cuda.manual_seed_all(_seed_)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(_seed_)


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    epoch: int,
    dist: bool = False,
    tb_writer: SummaryWriter = None,
    use_sj: bool = False,
):

    torch.cuda.empty_cache()
    model.train()

    top1_correct = 0
    top5_correct = 0
    n_samples = 0
    total_loss = 0

    # progress bar
    if is_master():
        process_bar = tqdm.tqdm(total=len(data_loader))

    for step, (data, label) in enumerate(data_loader):
        input = data.transpose(0, 1).cuda(non_blocking=True)  # input.shape = (T, B, C, H, W)
        target = label.cuda(non_blocking=True)                # target.shape = (B,)
        
        output = model(input)                                 # output.shape = (B, num_classes)
        loss = criterion(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if use_sj:
            functional.reset_net(model)

        # calculate the top5 and top1 accurate numbers
        _, predicted = output.cpu().topk(5, 1, True, True)
        top1_correct += predicted[:, 0].eq(label).sum().item()
        top5_correct += predicted.T.eq(label[None]).sum().item()
        total_loss += loss.item() * len(label)
        n_samples += len(label)

        if is_master():
            tb_writer.add_scalar('step/loss', loss.item(), epoch * len(data_loader) + step)
            process_bar.update(1)
    
    if dist:
        top1_correct, top5_correct, total_loss = global_meters_sum(top1_correct, top5_correct, total_loss)
    top1_acc = top1_correct / n_samples
    top5_acc = top5_correct / n_samples

    if is_master():
        if tb_writer is not None:  
            tb_writer.add_scalar('train/acc@1', top1_acc, epoch + 1)
            tb_writer.add_scalar('train/acc@5', top5_acc, epoch + 1)
            tb_writer.add_scalar('train/loss', total_loss / n_samples, epoch + 1)
        process_bar.close()

    print('train: acc@1: {:.5f}, acc@5: {:.5f}, loss: {:.6f}, cor@1: {}, cor@5: {}, total: {}'.format(
         top1_acc, top5_acc, total_loss / n_samples, top1_correct, top5_correct, n_samples
        )
    )
    

def validate(
    model: nn.Module,
    criterion: nn.Module,
    data_loader: DataLoader,
    epoch: int,
    dist: bool = False,
    tb_writer: SummaryWriter = None,
    use_sj: bool = False,
):
    torch.cuda.empty_cache()
    model.eval()

    top1_correct = 0
    top5_correct = 0
    n_samples = 0
    total_loss = 0

    # progress bar
    if is_master():
        process_bar = tqdm.tqdm(total=len(data_loader))

    with torch.no_grad():
        for step, (data, label) in enumerate(data_loader):
            input = data.transpose(0, 1).cuda(non_blocking=True)
            target = label.cuda(non_blocking=True)

            output = model(input)
            loss = criterion(output, target)
            
            if use_sj:
                functional.reset_net(model)

            # calculate the top5 and top1 accurate numbers
            _, predicted = output.cpu().topk(5, 1, True, True)  # batch_size, topk(5) 
            top1_correct += predicted[:, 0].eq(label).sum().item()
            top5_correct += predicted.T.eq(label[None]).sum().item()
            total_loss += loss.item() * len(label)
            n_samples += len(label)
            if is_master():
                process_bar.update(1)
    
    if dist:
        top1_correct, top5_correct, total_loss = global_meters_sum(top1_correct, top5_correct, total_loss)
    top1_acc = top1_correct / n_samples
    top5_acc = top5_correct / n_samples
    if is_master():
        if tb_writer is not None:
            tb_writer.add_scalar('val/acc@1', top1_acc, epoch + 1)
            tb_writer.add_scalar('val/acc@5', top5_acc, epoch + 1)
            tb_writer.add_scalar('val/loss', total_loss / n_samples, epoch + 1)
        process_bar.close()

    print('val: acc@1: {:.5f}, acc@5: {:.5f}, loss: {:.6f}, cor@1: {}, cor@5: {}, total: {}'.format(    
         top1_acc, top5_acc, total_loss / n_samples, top1_correct, top5_correct, n_samples
        )
    )

    return top1_acc


def train(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    train_loader: DataLoader,
    val_loader: DataLoader,
    nepochs: int,
    epoch: int,
    output_dir: str,
    args: argparse.Namespace,
    use_sj: bool = False,
):  
    tb_writer = None
    if is_master():
        tb_writer = SummaryWriter(output_dir + '/log')
    print('Save log to {}'.format(output_dir + '/log'))

    torch.cuda.empty_cache()
    
    while(epoch < nepochs):
        print('Epoch [{}/{}]'.format(epoch+1, nepochs))
        train_one_epoch(model, criterion, optimizer, train_loader, epoch, args.distributed, tb_writer, use_sj)
        val_acc = validate(model, criterion, val_loader, epoch, args.distributed, tb_writer, use_sj)
        
        epoch += 1
        scheduler.step()

        if epoch % args.save_freq == 0:
            checkpoint = {
                'model': model.module.state_dict() if args.distributed else model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
            }
            save_name = 'checkpoints/ckpt_epoch{}_acc{:.2f}.pth'.format(epoch, val_acc)
            save_on_master(checkpoint, os.path.join(output_dir, save_name))
            print('Saved checkpoint to [{}]'.format(output_dir))