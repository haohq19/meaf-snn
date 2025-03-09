# Cache pseudo labels / pseudo logits / feature maps for transferability assessment

import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import tqdm
from torch.utils.data import DataLoader
import models.spiking_resnet_event as spiking_resnet
import models.sew_resnet_event as sew_resnet
import models.ma_snn.Att_SNN as att_snn
import models.spiking_mlp_event as spiking_mlp
from spikingjelly.activation_based import functional
from utils.data import get_event_data_loader

def parser_args():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--dataset', default='n_mnist', type=str, help='dataset')
    parser.add_argument('--num_classes', default=10, type=int, help='number of classes')
    parser.add_argument('--nsteps', default=8, type=int, help='number of time steps')
    parser.add_argument('--connect_f', default='ADD', type=str, help='spike-element-wise connect function')
    parser.add_argument('--device_id', default=0, type=int, help='GPU id to use.')
    parser.add_argument('--nworkers', default=32, type=int, help='number of workers')
    parser.add_argument('--pt_dir', default='weights/event', help='path to pretrained weights')
    parser.add_argument('--output_dir', default='outputs', help='path where to save')
    return parser.parse_args()


def cache_representations(
    model: nn.Module,
    data_loader: DataLoader,
    cache_dir: str,
    use_spikingjelly: bool = True,
):  
    if os.path.exists(cache_dir):
        print('Cache already exists')
        return
    else:
        os.makedirs(cache_dir)
        print('Cache not found, make cache directory at [{}]'.format(cache_dir))
    
    with torch.no_grad():
        model.eval()
        features = []
        logits = []
        labels = []
        nsteps_per_epoch = len(data_loader)
        process_bar = tqdm.tqdm(total=nsteps_per_epoch)
        for data, label in data_loader:
            if len(data.shape) == 5:                                    # check event-based data shape: (B, T, C, H, W)
                input = data.cuda(non_blocking=True).transpose(0, 1)    # input.shape = (T, B, C, H, W)
            else:
                raise ValueError('Invalid data shape')

            label = label.numpy()                                       # label.shape = (B,)
            output = model(input)                                       # output.shape = (B, num_classes)
            
            feature_map = model.feature.detach().cpu().numpy()          # feature_map.shape = (T, B, num_features)
            features.append(feature_map)
            logit = output.softmax(dim=1).detach().cpu().numpy()        # logit.shape = (B, num_classes)
            logits.append(logit)
            labels.append(label)
            
            if use_spikingjelly:
                functional.reset_net(model)
            
            process_bar.update(1)
        process_bar.close()
        
        features = np.concatenate(features, axis=1).transpose(1, 0, 2)  # features.shape = (T, nsamples, num_features) -> (nsamples, T, num_features)
        logits = np.concatenate(logits, axis=0)                         # logits.shape = (nsamples, num_classes)
        labels = np.concatenate(labels, axis=0)                         # labels.shape = （nsamples,）
        np.save(os.path.join(cache_dir, 'features.npy'), features)
        np.save(os.path.join(cache_dir, 'logits.npy'), logits)
        np.save(os.path.join(cache_dir, 'labels.npy'), labels)


def load_pretrained_model(args):
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
            model.load_state_dict(state_dict)
            print('Load pretrained weights from [{}]'.format(pt_path))
    elif args.model in spiking_resnet.__dict__:
        model = spiking_resnet.__dict__[args.model](num_classes=1000, T=args.nsteps)
        if pt_weights is not None:
            state_dict = {k.replace('module.', ''): v for k, v in pt_weights.items()}
            model.load_state_dict(state_dict)
            print('Load pretrained weights from [{}]'.format(pt_path))
    elif args.model in att_snn.__dict__:
        model = att_snn.__dict__[args.model](num_classes=1000, T=args.nsteps)
        if pt_weights is not None:
            state_dict = {k.replace('module.', ''): v for k, v in pt_weights.items()}
            model.load_state_dict(state_dict)
            print('Load pretrained weights from [{}]'.format(pt_path))
    elif args.model in spiking_mlp.__dict__:
        model = spiking_mlp.__dict__[args.model](num_classes=1000, T=args.nsteps)
        if pt_weights is not None:
            state_dict = {k.replace('module.', ''): v for k, v in pt_weights.items()}
            model.load_state_dict(state_dict)
            print('Load pretrained weights from [{}]'.format(pt_path))

    else:
        raise NotImplementedError(args.model)
    return model


def main(args, data_loader, dataset_type):
    
    # device
    torch.cuda.set_device(args.device_id)

    # model
    model = load_pretrained_model(args)
    model.cuda()
    
    # cache representations
    cache_dir = os.path.join(args.output_dir, 'cache', args.dataset, args.model, dataset_type)  # output_dir/dataset/model/cache
    
    if model.__class__.__name__ == 'Net':
        use_spikingjelly = False
    else:
        use_spikingjelly = True

    if not os.path.exists(cache_dir):
        cache_representations(
            model=model,
            data_loader=data_loader,
            cache_dir=cache_dir,
            use_spikingjelly=use_spikingjelly,
        )


if __name__ == '__main__':
    models = ['spiking_mlp12', 'att_snn', 'sew_resnet18', 'sew_resnet34', 'sew_resnet50', 'sew_resnet101', 'sew_resnet152', 'spiking_resnet18', 'spiking_resnet34', 'spiking_resnet50']
    args = parser_args()
    train_loader, _, _ = get_event_data_loader(args)
    for model in models:
        args.model = model
        main(args, train_loader, 'train')