import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from sklearn.cluster import KMeans

from Metrics import nmi, acc

class DSSL(nn.Module):
    def __init__(self, AE, data_loader, dataset_size, batch_size=500, pretraining_epoch =200, MaxIter = 200, num_cluster = 10, m = 1.5, T=2, latent_size = 10, dataset_name = 'MNIST'):
        super(DSSL, self).__init__()
        self.AE = AE
        self.u_mean = torch.zeros([num_cluster,latent_size])
        self.batch_size = batch_size
        self.pretraining_epoch = pretraining_epoch
        self.MaxIter = MaxIter
        self.num_cluster = num_cluster
        self.data_loader = data_loader
        self.dataset_size = dataset_size
        self.m = m
        self.T=T
        self.latent_size = latent_size
        self.dataset_name = dataset_name


    def Kmeans_model_evaluation(self):
        self.AE.eval()
        datas = np.zeros([self.dataset_size, self.latent_size])
        label_true = np.zeros(self.dataset_size)
        ii = 0
        for x, target in self.data_loader:
            x = Variable(x).cuda()

            _, u = self.AE(x)
            u = u.cpu()
            datas[ii * self.batch_size:(ii + 1) * self.batch_size, :] = u.data.numpy()
            label_true[ii * self.batch_size:(ii + 1) * self.batch_size] = target.numpy()
            ii = ii + 1
        kmeans = KMeans(n_clusters=self.num_cluster, random_state=0).fit(datas)

        label_pred = kmeans.labels_
        ACC = acc(label_true, label_pred, self.num_cluster)
        print('ACC', ACC)
        print('NMI', nmi(label_true, label_pred))
        return ACC


    def pretrain(self):
        self.AE.train()
        self.AE.cuda()
        for param in self.AE.parameters():
            param.requires_grad = True
        optimizer = optim.Adam(self.AE.parameters())
        prev_ACC = 0
        for T in range(0, self.pretraining_epoch):
            print('Pretraining Iteration: ', T + 1)
            for x, target in self.data_loader:
                optimizer.zero_grad()
                x = Variable(x).cuda()
                y, _ = self.AE(x)
                loss = nn.MSELoss()(x, y)
                loss.backward()
                optimizer.step()

            ACC = self.Kmeans_model_evaluation()
            if ACC > prev_ACC:
                prev_ACC = ACC
                with open('AE_'+self.dataset_name+'_pretrain', 'wb') as f:
                    torch.save(self.AE, f)

        self.AE = torch.load('AE_'+self.dataset_name+'_pretrain')
        return self.AE

    def initialization(self):
        print("-----initialization mode--------")
        self.AE = torch.load('AE_'+self.dataset_name+'_pretrain')
        self.AE.cuda()
        datas = np.zeros([self.dataset_size, self.latent_size])
        ii = 0
        for x, target in self.data_loader:
            x = Variable(x).cuda()
            _, u = self.AE(x)
            u = u.cpu()
            datas[(ii) * self.batch_size:(ii + 1) * self.batch_size] = u.data.numpy()
            ii = ii + 1
        # datas = datas.cpu()
        kmeans = KMeans(n_clusters=self.num_cluster, random_state=0).fit(datas)
        self.u_mean = kmeans.cluster_centers_
        self.u_mean = torch.from_numpy(self.u_mean)
        self.u_mean = Variable(self.u_mean).cuda()
        return self.AE, self.u_mean

    def cmp(self, u, u_mean):
        p = torch.zeros([self.batch_size, self.num_cluster]).cuda()
        for j in range(0, self.num_cluster):
            p[:, j] = torch.sum(torch.pow(u[j, :, :] - u_mean[j, :].unsqueeze(0).repeat(self.batch_size, 1), 2), dim=1)
        p = torch.pow(p, -1 / (self.m - 1))
        sum1 = torch.sum(p, dim=1)
        p = torch.div(p, sum1.unsqueeze(1).repeat(1, self.num_cluster))
        # print(p[1,:])
        return p

    def clustering_cost(self, x, y, u, p, u_means):
        return torch.matmul(p, torch.sum(torch.pow(x - y, 2), dim=1)) + 0.1* torch.matmul(p, torch.sum(torch.pow(u - u_means, 2), dim=1))


    def update_cluster_centers(self):
        self.AE.eval()
        for param in self.AE.parameters():
            param.requires_grad = False
        den = torch.zeros([self.num_cluster]).cuda()
        num = torch.zeros([self.num_cluster, self.latent_size]).cuda()
        for x, target in self.data_loader:
            x = Variable(x).cuda()
            _, u = self.AE(x)
            p = self.cmp(u.unsqueeze(0).repeat(self.num_cluster,1,1),self.u_mean)
            p = torch.pow(p, self.m)
            for kk in range(0, self.num_cluster):
                den[kk] = den[kk] + torch.sum(p[:, kk])
                num[kk, :] = num[kk, :] + torch.matmul(p[:, kk].T, u)
        for kk in range(0, self.num_cluster):
            self.u_mean[kk, :] = torch.div(num[kk, :], den[kk])
        self.AE.cuda()
        self.AE.train()
        for param in self.AE.parameters():
            param.requires_grad = True
        return self.u_mean


    def model_evaluation(self):
        pred_labels = np.zeros(self.dataset_size)
        true_labels = np.zeros(self.dataset_size)
        ii = 0
        for x, target in self.data_loader:
            x = Variable(x).cuda()
            _, u = self.AE(x)

            u = u.unsqueeze(0).repeat(self.num_cluster,1,1)
            p = self.cmp(u,self.u_mean)
            y = torch.argmax(p, dim=1)
            y = y.cpu()
            y = y.numpy()
            pred_labels[(ii) * self.batch_size:(ii + 1) * self.batch_size] = y
            true_labels[(ii) * self.batch_size:(ii + 1) * self.batch_size] = target.numpy()
            ii = ii + 1
        ACC = acc(true_labels, pred_labels, self.num_cluster)
        NMI = nmi(true_labels,pred_labels)
        print('ACC', ACC)
        print('NMI', NMI)

        self.AE.cuda()
        self.AE.train()
        for param in self.AE.parameters():
            param.requires_grad = True

        return ACC, NMI


    def train(self):
      self.AE, self.u_mean = self.initialization()
      self.AE.cuda()
      self.AE.train()
      for param in self.AE.parameters():
          param.requires_grad = True
      optimizer = optim.SGD(self.AE.parameters(), lr=0.000001, momentum=0.9)
      ACC_prev = 0.0

      for T in range(0,self.MaxIter):
        print('Iteration: ', T + 1)
        if T% self.T ==1:
            self.u_mean = self.update_cluster_centers()
        for x, target in self.data_loader:
            u = torch.zeros([self.num_cluster, self.batch_size, self.latent_size]).cuda()
            x = Variable(x).cuda()
            for kk in range(0, self.num_cluster):
                _, y = self.AE(x)
                u[kk, :, :] = y.cuda()
            u = u.detach()
            p = self.cmp(u, self.u_mean.cuda())
            p = p.detach()
            self.u_mean = self.u_mean.cuda()
            p = p.T
            p = torch.pow(p, self.m)
            for i in range(0, self.num_cluster):
                    y, u1 = self.AE(x)
                    self.u_mean = self.u_mean.float()
                    loss = self.clustering_cost(x.view(-1, 784), y.view(-1, 784), u1, p[i, :].unsqueeze(0), self.u_mean[i, :].unsqueeze(0).repeat(self.batch_size, 1))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        ACC, NMI = self.model_evaluation()
        if ACC > ACC_prev:
            ACC_prev = ACC
            with open('AE_result_'+self.dataset_name, 'wb') as f:
                torch.save(self.AE, f)
            with open('Centers_'+self.dataset_name, 'wb') as f:
                torch.save(self.u_mean, f)




