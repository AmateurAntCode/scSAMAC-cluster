import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from layers import NBLoss, MeanAct, DispAct
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import math, os
import st_loss
from sklearn import metrics
from single_cell_tools import cluster_acc
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
import random
def buildNetwork(layers, type, activation="relu"):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        if activation=="relu":
            net.append(nn.ReLU())
        elif activation=="sigmoid":
            net.append(nn.Sigmoid())
    return nn.Sequential(*net)

def euclidean_dist(x, y):
    return torch.sum(torch.square(x - y), dim=1)
class Generate_Model(torch.nn.Module):
    '''
    生成器
    '''

    def __init__(self, input_dim):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=128, out_features=256),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=256, out_features=512),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=512, out_features=input_dim),
            torch.nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class Distinguish_Model(torch.nn.Module):
    '''
    判别器
    '''

    def __init__(self, input_dim):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=input_dim, out_features=512),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=512, out_features=256),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=256, out_features=128),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=128, out_features=1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        return x
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()  # 继承自nn.Module基类
        self.n_heads = n_heads  # 多头注意力头数
        self.d_model = d_model  # 输入向量维度
        self.d_k = d_model // n_heads  # 每个头的维度
        self.dropout = nn.Dropout(p=dropout)  # dropout概率

        # 初始化Query、Key、Value的权重矩阵
        self.W_q = nn.Linear(d_model, n_heads * self.d_k)  # Query权重矩阵
        self.W_k = nn.Linear(d_model, n_heads * self.d_k)  # Key权重矩阵
        self.W_v = nn.Linear(d_model, n_heads * self.d_k)  # Value权重矩阵

        # 初始化输出的权重矩阵
        self.W_o = nn.Linear(n_heads * self.d_k, d_model)  # 输出向量的权重矩阵

    def forward(self, x, mask=None):
        # 输入 x 的维度为 [batch_size, seq_len, d_model]
        batch_size, seq_len, d_model = x.size()

        # 通过权重矩阵计算 Q、K、V
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k)

        # 交换维度以便于计算注意力权重
        Q = Q.permute(0, 2, 1, 3).contiguous().view(batch_size * self.n_heads, seq_len, self.d_k)
        K = K.permute(0, 2, 1, 3).contiguous().view(batch_size * self.n_heads, seq_len, self.d_k)
        V = V.permute(0, 2, 1, 3).contiguous().view(batch_size * self.n_heads, seq_len, self.d_k)

        # 计算注意力权重
        scores = torch.bmm(Q, K.transpose(1, 2)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = nn.Softmax(dim=-1)(scores)
        attn_weights = self.dropout(attn_weights)

        # 计算输出向量
        attn_output = torch.bmm(attn_weights, V)
        attn_output = attn_output.view(batch_size, self.n_heads, seq_len, self.d_k)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len,
                                                                        self.n_heads * self.d_k)
        output = self.W_o(attn_output)

        return output




class scSAMAC(nn.Module):
    def __init__(self, input_dim, z_dim, encodeLayer=[], decodeLayer=[], 
            activation="relu", sigma=1., alpha=1., gamma=1., device="cuda"):
        super(scSAMAC, self).__init__()
        torch.set_default_dtype(torch.float64)
        self.z_dim = z_dim
        self.activation = activation
        self.sigma = sigma
        self.alpha = alpha
        self.input_dim = input_dim
        self.gamma = gamma
        self.device = device
        self.encoder = buildNetwork([input_dim]+encodeLayer, type="encode", activation=activation)
        self.decoder = buildNetwork([z_dim]+decodeLayer, type="decode", activation=activation)
        self._enc_mu = nn.Linear(encodeLayer[-1], z_dim)
        self._dec_mean = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), MeanAct())
        self._dec_disp = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), DispAct())
        #self._dec_pi = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), nn.Sigmoid())

        self.nb_loss = NBLoss().to(self.device)

        self.mutiattention = MultiHeadAttention(8, 64).to(self.device)
        self.mutiattention2 = MultiHeadAttention(8, 32).to(self.device)
        #self.mutiattention3 = MultiHeadAttention(8, 11).to(self.device)
        self.to(device)

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)

    def soft_assign(self, z):
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.mu)**2, dim=2) / self.alpha)
        q = q**((self.alpha+1.0)/2.0)
        q = (q.t() / torch.sum(q, dim=1)).t()
        return q
    
    def target_distribution(self, q):
        p = q**2 / q.sum(0)
        return (p.t() / p.sum(1)).t()

    def forwardAE(self, x):
        h = self.encoder(x + torch.randn_like(x) * self.sigma)
        h = self.attentionblock1(h)

        z = self._enc_mu(h)
        z = self.attentionblock2(z)

        h = self.decoder(z)
        _mean = self._dec_mean(h)
        _disp = self._dec_disp(h)
        #_pi = self._dec_pi(h)


        h0 = self.encoder(x)
        h0 = self.attentionblock1(h0)

        z0 = self._enc_mu(h0)
        z0 = self.attentionblock2(z0)
        return z0, _mean, _disp

    def forwardAE_neg(self, x):
        h0 = self.encoder(x)
        h0 = self.attentionblock1(h0)
        z0 = self._enc_mu(h0)
        z0 = self.attentionblock2(z0)
        return z0

    def forward(self, x):
        h = self.encoder(x + torch.randn_like(x) * self.sigma)
        h = self.attentionblock1(h)

        z = self._enc_mu(h)
        z = self.attentionblock2(z)

        h = self.decoder(z)
        _mean = self._dec_mean(h)
        _disp = self._dec_disp(h)
        #_pi = self._dec_pi(h)

        h0 = self.encoder(x)
        h0 = self.attentionblock1(h0)

        z0 = self._enc_mu(h0)
        z0 = self.attentionblock2(z0)

        q = self.soft_assign(z0)
        return z0, q, _mean, _disp
    def attentionblock1(self, x):
        temp = x
        x = x.unsqueeze(0)
        x = self.mutiattention(x)
        x = x.view(-1, x.size(-1))
        return x + temp

    def attentionblock2(self, x):
        temp = x
        x = x.unsqueeze(0)
        x = self.mutiattention2(x)
        x = x.view(-1, x.size(-1))
        return x + temp

    def calculate_neighbors(self, data, n_neighbors=10):
        # 使用KNN算法计算近邻
        data = data.cpu().numpy()
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(data)
        distances, indices = nbrs.kneighbors(data)
        return indices

    def encodeBatch(self, X, batch_size=256):
        self.eval()
        encoded = []
        num = X.shape[0]
        num_batch = int(math.ceil(1.0*X.shape[0]/batch_size))
        for batch_idx in range(num_batch):
            xbatch = X[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
            inputs = Variable(xbatch).to(self.device)
            z, _, _= self.forwardAE(inputs)
            encoded.append(z.data)

        encoded = torch.cat(encoded, dim=0)
        return encoded.to(self.device)
    def intra_cluster_distance(self, cluster):
        # 计算簇内平均距离
        distances = torch.pdist(cluster)
        return distances.mean()

    def inter_cluster_distance(self, cluster1, cluster2):
        # 计算簇间平均距离
        distances = torch.cdist(cluster1, cluster2)
        return distances.mean()

    def soft_k_loss(self, z_latent):
        dist1 = torch.sum((z_latent.unsqueeze(1) - self.mu).pow(2), dim=2)
        min_values = torch.min(dist1, dim=1)[0]
        temp_dist1 = dist1 - min_values.view(-1, 1)
        q = torch.exp(-temp_dist1)
        q = (q.t() / torch.sum(q, dim=1)).t()
        q = q**2
        q = (q.t() / torch.sum(q, dim=1)).t()
        dist2 = dist1 * q
        return torch.mean(torch.sum(dist2, dim=1))

    def wasserstein_distance(self, p, q):
        from scipy.stats import wasserstein_distance
        return wasserstein_distance(p.detach().cpu().numpy().flatten(), q.detach().cpu().numpy().flatten())


    def cluster_loss(self, p, q):
        def kld(target, pred):
            return torch.mean(torch.sum(target*torch.log(target/(pred+1e-6)), dim=-1))
        kldloss = kld(p, q)
        return self.gamma*kldloss


    def unsupervised_contrastive_loss(self, matrix1, matrix2, margin=1.0):

        matrix1 = matrix1.cpu().detach().numpy()
        matrix2 = matrix2.cpu().detach().numpy()
        m, n = matrix1.shape
        matrix1 = torch.tensor(matrix1)
        matrix2 = torch.tensor(matrix2)
        loss = 0.0

        # 计算正样本对的损失
        for i in range(n):
            col1 = matrix1[:, i]
            col2 = matrix2[:, i]

            # 计算欧几里得距离
            positive_distance = torch.norm(col1 - col2)
            loss += positive_distance ** 2

        # 计算负样本对的损失
        for i in range(n):
            for j in range(n):
                if i != j:
                    col1 = matrix1[:, i]
                    col2 = matrix2[:, j]

                    # 计算欧几里得距离
                    negative_distance = torch.norm(col1 - col2)
                    loss += torch.clamp(margin - negative_distance, min=0.0) ** 2

        # 返回平均损失
        return loss / (n * n)

    def process_matrix(self, matrix):
        # Convert the input matrix to a numpy array for easier manipulation
        matrix = matrix.cpu().numpy()
        matrix = np.array(matrix)

        # Step 1: Find the column with the maximum sum
        col_sums = matrix.sum(axis=0)
        max_col_sum = np.max(col_sums)

        # Step 2: Process each column based on the given probabilities
        for i, col_sum in enumerate(col_sums):
            ratio = col_sum / max_col_sum

            if ratio < 0.1:
                if np.random.rand() < 0.6:
                    matrix[:, i] = 0
            elif ratio < 0.3:
                if np.random.rand() < 0.3:
                    matrix[:, i] = 0
            else:
                if np.random.rand() < 0.1:
                    matrix[:, i] = 0

        return torch.tensor(matrix)

    def pretrain_autoencoder(self, X, X_raw, size_factor, batch_size=256, lr=0.0001, epochs=400, ae_save=True, ae_weights='AE_weights.pth.tar'):
        self.train()
        dataset = TensorDataset(torch.Tensor(X), torch.Tensor(X_raw), torch.Tensor(size_factor))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        criterion_rep = st_loss.SupConLoss(temperature=0.07)
        #D_optim = torch.optim.Adam(self.D.parameters(), lr=1e-4)
        #G_optim = torch.optim.Adam(self.G.parameters(), lr=1e-4)
        print("Pretraining stage")
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, amsgrad=True)
        for epoch in range(epochs):
            loss_val = 0
            for batch_idx, (x_batch, x_raw_batch, sf_batch) in enumerate(dataloader):
                x_tensor = Variable(x_batch).to(self.device)
                neg_sample = self.process_matrix(x_tensor).to(self.device)
                out1 = self.forwardAE_neg(neg_sample)

                x_raw_tensor = Variable(x_raw_batch).to(self.device)
                sf_tensor = Variable(sf_batch).to(self.device)
                z_tensor, mean_tensor, disp_tensor = self.forwardAE(x_tensor)

                features2 = torch.cat(
                    [out1.unsqueeze(1),
                     z_tensor.unsqueeze(1)],
                    dim=1).to(self.device)
                contrast_loss2 = criterion_rep(features2)

                loss = self.nb_loss(x=x_raw_tensor, mean=mean_tensor, disp=disp_tensor, scale_factor=sf_tensor) + contrast_loss2 * 0.01
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_val += loss.item() * len(x_batch)
            print('Pretrain epoch %3d, NB loss: %.8f' % (epoch+1, loss_val/X.shape[0]))

        if ae_save:
            torch.save({'ae_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, ae_weights)

    def save_checkpoint(self, state, index, filename):
        newfilename = os.path.join(filename, 'FTcheckpoint_%d.pth.tar' % index)
        torch.save(state, newfilename)

    def fit(self, X, X_raw, size_factor, n_clusters, init_centroid=None, y=None, y_pred_init=None, lr=1., batch_size=256,
            num_epochs=10, update_interval=1, tol=1e-3, save_dir=""):
        '''X: tensor data'''
        self.train()
        print("Clustering stage")
        #self.level_loss = contrastive_loss.ClusterLoss(n_clusters, temperature=1.0, device=self.device)

        level_optim = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=0.001, amsgrad=True)
        X = torch.tensor(X, dtype=torch.float64)
        X_raw = torch.tensor(X_raw, dtype=torch.float64)
        size_factor = torch.tensor(size_factor, dtype=torch.float64)
        self.mu = Parameter(torch.Tensor(n_clusters, self.z_dim).to(self.device))#聚类中心
        optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, rho=.95)
        print("Initializing cluster centers with kmeans.")
        if init_centroid is None:
            kmeans = KMeans(n_clusters, n_init=20)
            data = self.encodeBatch(X)
            self.y_pred = kmeans.fit_predict(data.data.cpu().numpy())
            self.y_pred_last = self.y_pred
            self.mu.data.copy_(torch.tensor(kmeans.cluster_centers_, dtype=torch.float64))
        else:
            self.mu.data.copy_(torch.tensor(init_centroid, dtype=torch.float64))
            self.y_pred = y_pred_init
            self.y_pred_last = self.y_pred
        if y is not None:
            acc = np.round(cluster_acc(y, self.y_pred), 5)
            nmi = np.round(metrics.normalized_mutual_info_score(y, self.y_pred), 5)
            ari = np.round(metrics.adjusted_rand_score(y, self.y_pred), 5)
            print('Initializing k-means: NMI= %.4f, ARI= %.4f, ACC= %.4f' % (nmi, ari, acc))

        num = X.shape[0]
        num_batch = int(math.ceil(1.0*X.shape[0]/batch_size))

        final_acc, final_nmi, final_ari, final_epoch = 0, 0, 0, 0
        train_losses = []

        for epoch in range(num_epochs):
            if epoch%update_interval == 0:
                # update the targe distribution p
                latent = self.encodeBatch(X.to(self.device))
                q = self.soft_assign(latent)
                p = self.target_distribution(q).data

                # evalute the clustering performance
                self.y_pred = torch.argmax(q, dim=1).data.cpu().numpy()

                if y is not None:
                    final_acc = acc = np.round(cluster_acc(y, self.y_pred), 5)
                    final_nmi = nmi = np.round(metrics.normalized_mutual_info_score(y, self.y_pred), 5)
                    final_epoch = ari = np.round(metrics.adjusted_rand_score(y, self.y_pred), 5)
                    print('Clustering   %d: NMI= %.4f, ARI= %.4f, ACC= %.4f' % (epoch+1, nmi, ari, acc))

                # save current model
                if (epoch>0 and delta_label < tol) or epoch%10 == 0:
                    self.save_checkpoint({'epoch': epoch+1,
                            'state_dict': self.state_dict(),
                            'mu': self.mu,
                            'y_pred': self.y_pred,
                            'y_pred_last': self.y_pred_last,
                            'y': y
                            }, epoch+1, filename=save_dir)

                # check stop criterion
                delta_label = np.sum(self.y_pred != self.y_pred_last).astype(np.float32) / num
                self.y_pred_last = self.y_pred
                if epoch>0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print("Reach tolerance threshold. Stopping training.")
                    break


            # train 1 epoch for clustering loss
            train_loss = 0.0
            recon_loss_val = 0.0
            cluster_loss_val = 0.0
            #criterion_rep = st_loss.SupConLoss(temperature=0.07)
            for batch_idx in range(num_batch):
                xbatch = X[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                xrawbatch = X_raw[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                sfbatch = size_factor[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                pbatch = p[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                optimizer.zero_grad()
                level_optim.zero_grad()
                inputs = Variable(xbatch).to(self.device)
                rawinputs = Variable(xrawbatch).to(self.device)
                sfinputs = Variable(sfbatch).to(self.device)
                target = Variable(pbatch).to(self.device)

                neg_sample = self.process_matrix(inputs).to(self.device)
                #neg_sample = self.apply_multiple_random_masks(neg_sample).to(self.device)
                out1 = self.forwardAE_neg(neg_sample)

                zbatch, qbatch, meanbatch, dispbatch = self.forward(inputs)
                #out1 = self.cluster_level(out1, n_clusters).to(self.device)
                #temp_zbatch = self.cluster_level(zbatch, n_clusters).to(self.device)

                '''features = torch.cat(
                    [out1.t().unsqueeze(1),
                     temp_zbatch.t().unsqueeze(1)],
                    dim=1).to(self.device)
                contrast_loss = criterion_rep(features)
                level_loss = contrast_loss
                print(f'level loss is {level_loss}')'''
                #self.cluster_loss(target, qbatch)
                #self.wasserstein_distance(target, qbatch)

                cluster_loss = self.wasserstein_distance(target, qbatch) + self.soft_k_loss(zbatch) * 0.001 + self.unsupervised_contrastive_loss(self.cluster_level(zbatch, n_clusters), self.cluster_level(out1, n_clusters)) * 0.002          #print(f'level loss is {self.unsupervised_contrastive_loss(self.cluster_level(zbatch, n_clusters), self.cluster_level(out1, n_clusters))}')
                #soft_loss = self.soft_k_loss(zbatch)
                #entropy_loss = self.entropy_loss(target, qbatch)
                recon_loss = self.nb_loss(rawinputs, meanbatch, dispbatch, sfinputs)

                loss = cluster_loss*self.gamma + recon_loss
                #level_loss.backward(retain_graph=True)
                loss.backward()
                optimizer.step()
                #level_loss.step()
                cluster_loss_val += cluster_loss.item() * len(inputs)
                recon_loss_val += recon_loss.item() * len(inputs)
                train_loss += loss.item() * len(inputs)
            print("Epoch %3d: Total: %.8f Clustering Loss: %.8f NB Loss: %.8f" % (
                epoch + 1, train_loss / num, cluster_loss_val / num, recon_loss_val / num))
            train_losses.append(train_loss / num)
            plt.figure()
            plt.plot(range(epoch + 1), train_losses, 'b', label='Train Loss')
            #plt.title('Training Loss of cite')
            plt.xlabel('Epochs')
            plt.ylabel('Total loss')
            plt.legend()
            plt.savefig('train_loss_curve.png')  # 保存损失曲线图像
            plt.close()
        return self.y_pred, final_acc, final_nmi, final_ari, final_epoch

