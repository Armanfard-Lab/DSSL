
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms

from DSSL import DSSL
from Network import AutoEncoder

batch_size = 500
dataset_size = 70000
train_set = dset.MNIST(root='/home/mrsadeghi/Spectral_clustering_network', train=True,
                       transform=transforms.ToTensor(), download=True)
test_set = dset.MNIST(root='/home/mrsadeghi/Spectral_clustering_network', train=False,
                      transform=transforms.ToTensor(), download=True)

kwargs = {'num_workers': 1}

train1 = torch.utils.data.ConcatDataset([train_set, test_set])
data_loader = torch.utils.data.DataLoader(
    dataset=train1,
    batch_size=batch_size,
    shuffle=True, **kwargs)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias.data)

        # torch.nn.init.xavier_uniform(m.bias.data)
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias.data)



if __name__ == '__main__':
    AE = AutoEncoder().cuda()
    AE.apply(weights_init)


    batch_size = 500
    pretraining_epoch = 0
    MaxIter = 200
    num_cluster = 10
    m = 1.5
    T = 2
    latent_size = 10
    dataset_name = 'MNIST'

    DSSL = DSSL(AE, data_loader, dataset_size, batch_size=batch_size, pretraining_epoch =pretraining_epoch,
                MaxIter = MaxIter, num_cluster = num_cluster, m = m, T=T,  latent_size = latent_size,  dataset_name = dataset_name)

    if pretraining_epoch!= 0:
        DSSL.pretrain()
    if MaxIter != 0:
        DSSL.train()

