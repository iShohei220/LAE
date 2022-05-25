import argparse
import os
import shutil
import math
import random

import numpy as np
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision

from model import DLGM
from torchvision.datasets import MNIST, SVHN, CIFAR10, CelebA
from torchvision.utils import make_grid


def train(rank, model, optimizer,
          dataloader, writer, epoch, detect_anomaly=False):
    model.train()
    running_loss = 0
    running_rec = 0
    N = 0
    data_size = len(dataloader.dataset)
    with torch.autograd.set_detect_anomaly(detect_anomaly):
        for step, (x, y) in enumerate(dataloader):
            x = x.to(next(model.parameters()).device)
            optimizer.zero_grad()
            loss = model(x)
            logit_tau = model.module.logit_tau
            loss = loss.mean() + torch.sum(logit_tau + 2 * F.softplus(- logit_tau)) / data_size
            loss.backward()
            optimizer.step()

            running_loss += x.size(0) * loss.item()
            mu_x = model.module.reconstruct(x)
            running_rec += torch.pow(x - mu_x, 2).mean([1, 2, 3]).sum().item()

            N += x.size(0)

    running_loss /= N
    running_rec /= N

    if rank == 0:
        writer.add_scalar("loss/train", running_loss, epoch)
        writer.add_scalar("reconstruction_error/train", running_rec, epoch)
        writer.add_images("ground_truth/train", x[-8:], epoch)
        writer.add_images(f"reconstruction/train", mu_x[-8:], epoch)


def valid(rank, model, dataloader, writer, epoch):
    model.eval()
    running_loss = 0
    running_rec = 0
    N = 0
    if dataloader is not None:
        for x, y in dataloader:
            x = x.to(next(model.parameters()).device)
            loss = model(x)

            running_loss += loss.sum().item()
            mu_x = model.module.reconstruct(x)
            running_rec += torch.pow(x - mu_x, 2).mean([1, 2, 3]).sum().item()

            N += x.size(0)

        running_loss /= N
        running_rec /= N

        if epoch >= 0 and rank == 0:
            writer.add_scalar("loss/test", running_loss, epoch)
            writer.add_scalar("reconstruction_error/test", running_rec, epoch)

        if epoch <= 0 and rank == 0:
            writer.add_images("ground_truth/test", x[-8:], epoch)

        if rank == 0:
            writer.add_images(f"reconstruction/test", mu_x[-8:], epoch)

    if rank == 0:
        sample = model.module.sample(64)
        writer.add_image("sample", make_grid(sample, nrow=8), epoch)

    return running_loss


def get_args():
    parser = argparse.ArgumentParser(description="Langevin Autoencoder")
    parser.add_argument('--num_workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--seed', type=int, help='random seed (default: 0)', default=0)
    parser.add_argument("--dataset", type=str, choices=["MNIST", "SVHN", "CIFAR10", "CelebA"],
                        default="MNIST", help='dataset to be used (default: MNIST)')
    parser.add_argument("--nz", type=int, default=8,
                        help="latent dimensionality")
    parser.add_argument("--nh", type=int, default=1024,
                        help="number of feature maps (default: 64)")
    parser.add_argument("--size", type=int, default=28,
                        help="image size (default: 28)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="learning rate (default: 1e-4)")
    parser.add_argument("--epoch", type=int, default=50,
                        help="number of epochs (default: 2000)")
    parser.add_argument("--batch_size", type=int, default=100,
                        help="batch size (default: 100)")
    parser.add_argument("--world_size", type=int, default=torch.cuda.device_count())
    parser.add_argument("--port", type=int, default=12355)
    parser.add_argument("--log_dir", type=str)
    parser.add_argument("--valid", action='store_true')
    parser.add_argument("--detect_anomaly", action='store_true')
    parser.add_argument("--model_path", type=str, default='models')
    parser.add_argument("--save_model", action='store_true')
    args = parser.parse_args()

    return args


def set_seed(seed):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)


def setup(rank, args):
    port = args.port
    os.environ['MASTER_ADDR'] = 'localhost'
    while True:
        os.environ['MASTER_PORT'] = str(port)
        try:
            # initialize the process group
            dist.init_process_group("gloo", rank=rank, world_size=args.world_size)
        except RuntimeError:
            port += 1
        else:
            break

def cleanup():
    dist.destroy_process_group()


def run(rank, args):
    setup(rank, args)

    # Dataset
    if args.dataset == "MNIST":
        nx = 1
        if args.size == 28:
            training_data = MNIST("~/dataset", download=True,
                             transform=torchvision.transforms.ToTensor())
            test_data = MNIST("~/dataset", train=False, download=True,
                            transform=torchvision.transforms.ToTensor())
        elif args.size == 32:
            training_data = MNIST("~/dataset", download=True,
                             transform=torchvision.transforms.Compose([
                                 torchvision.transforms.Pad(2),
                                 torchvision.transforms.ToTensor()]))
            test_data = MNIST("~/dataset", train=False, download=True,
                            transform=torchvision.transforms.Compose([
                                torchvision.transforms.Pad(2),
                                torchvision.transforms.ToTensor()]))
        else:
            raise NotImplementedError
    elif args.dataset == "SVHN":
        nx = 3
        if not args.size == 32:
            raise NotImplementedError
        training_data = SVHN("~/dataset/SVHN", split='train', download=True,
                        transform=torchvision.transforms.ToTensor())
        test_data = SVHN("~/dataset/SVHN", split='test', download=True,
                       transform=torchvision.transforms.ToTensor())
    elif args.dataset == "CIFAR10":
        nx = 3
        if args.size != 32:
            raise NotImplementedError
        training_data = CIFAR10("~/dataset/CIFAR10", download=True,
                           transform=torchvision.transforms.ToTensor())
        test_data = CIFAR10("~/dataset/CIFAR10", train=False, download=True,
                          transform=torchvision.transforms.ToTensor())
    elif args.dataset == "CelebA":
        nx = 3
        if args.size > 64:
            raise NotImplementedError
        training_data = CelebA("~/dataset/CelebA", 'train', download=True,
                          transform=torchvision.transforms.Compose([
                              torchvision.transforms.Resize(args.size),
                              torchvision.transforms.CenterCrop(args.size),
                              torchvision.transforms.ToTensor()]))
        test_data = CelebA("~/dataset/CelebA", 'test', download=True,
                         transform=torchvision.transforms.Compose([
                             torchvision.transforms.Resize(args.size),
                             torchvision.transforms.CenterCrop(args.size),
                             torchvision.transforms.ToTensor()]))

    kwargs = {'num_workers': args.num_workers, 'pin_memory': True}
    train_sampler = DistributedSampler(training_data, num_replicas=args.world_size, rank=rank, shuffle=True)
    train_dataloader = DataLoader(training_data, batch_size=args.batch_size, sampler=train_sampler, **kwargs)

    test_sampler = DistributedSampler(test_data, num_replicas=args.world_size, rank=rank)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, sampler=test_sampler, **kwargs)

    # Model
    model = DLGM(args.size, nx, args.nh, args.nz).to(rank)
    params = model.parameters()

    model = DDP(model, device_ids=[rank])

    optimizer = SGD(params, lr=args.lr)

    if args.log_dir is None:
        log_dir = f"{args.dataset}-{args.size}-DLGM-nh{args.nh}-nz{args.nz}-lr{args.lr}-epoch{args.epoch}-seed{args.seed}"
    else:
        log_dir = args.log_dir

    writer = None
    if rank == 0:
        # Tensorboard
        writer = SummaryWriter(os.path.join('../../runs', log_dir))

        # make a diectory for saving models
        if args.save_model:
            os.makedirs(os.path.join(args.model_path, log_dir), exist_ok=True)

    for epoch in tqdm(range(1, args.epoch+1)):
        train_sampler.set_epoch(epoch)
        train(rank, model, optimizer,
              train_dataloader, writer,
              epoch, args.detect_anomaly)
        if args.valid:
            loss = valid(rank, model, test_dataloader, writer, epoch)
        # Save model
        if args.save_model and rank == 0:
            torch.save(model.state_dict(), os.path.join(args.model_path, log_dir, f"model-{epoch}.pt"))
            if epoch > 1:
                os.remove(os.path.join(args.model_path, log_dir, f"model-{epoch-1}.pt"))

    loss = valid(rank, model, test_dataloader, writer, -1)
    if rank == 0:
        # add hyperparameters and final results to tensorboard
        writer.add_hparams({"model": "VAE", "dataset": args.dataset, "size": args.size,
                            "nz": args.nz, "nh": args.nh,
                            "lr": args.lr, "batch_size": args.batch_size,
                            "epoch": args.epoch, "seed": args.seed},
                            {"loss": loss})
        writer.close()


if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)

    mp.spawn(run,
        args=(args,),
        nprocs=args.world_size,
        join=True)
