import pickle
import math
import h5py
import numpy as np
import torch
import os
import torch.nn.functional as F
import torch.nn as nn
from scipy.stats import t
import scipy
from torch.nn import Parameter

use_gpu = torch.cuda.is_available()

# seed = 1
# torch.manual_seed(seed)  # 为CPU设置随机种子
# torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
# torch.cuda.manual_seed_all(seed)
# random.seed(seed)
# np.random.seed(seed)
# os.environ['PYTHONHASHSEED'] = str(seed)

def distribution_calibration(query, base_means, base_cov, all_features, k, k1=2000, alpha=0.21, use_smp=False, ratio=0.5):

    if use_smp:
        all_features_rep = all_features.reshape(-1, 640)

        cos_dis = torch.mm(F.normalize(query, dim=0)[None,:], F.normalize(all_features_rep, dim=1).transpose(1,0))
        cos_simi, cos_index = cos_dis[0].topk(2000, 0)
        cos_weight = torch.pow(cos_simi, 0.9) + 0.3

        select_means = (cos_weight[:, None] * all_features_rep[cos_index, :])
        calibrated_mean = torch.cat((query[None, :], select_means), dim=0).mean(dim=0)

        calibrated_cov = torch.cov((all_features_rep[cos_index, :]).T)
        return calibrated_mean, calibrated_cov

    else:

        cos_dis = torch.cdist(query[None, :], base_means, p=2)
        cos_simi, cos_index = (-cos_dis[0]).topk(2, 0)

        cos_weight = 1 / torch.pow(-cos_simi, 0.5) + 0.3
        cos_weight[cos_weight > 1] = 1

        select_means = (cos_weight[:, None] * base_means[cos_index, :])
        # calibrated_mean = (select_means + query[None, :]).squeeze()
        calibrated_mean = torch.cat((query[None, :], select_means), dim=0).mean(dim=0)

        calibrated_cov = (base_cov[cos_index, :]).mean(dim=0)

        return calibrated_mean, calibrated_cov

def recons_cov(orig_cov, eig_num):
    # eigval, eigvec = torch.symeig(orig_cov, eigenvectors=True)
    eigval, eigvec = torch.linalg.eigh(orig_cov)
    sort_idx = torch.argsort(eigval, descending=True)
    sort_eigval = eigval[sort_idx]
    eigvec = eigvec[:,sort_idx]
    recon_cov = torch.matmul(torch.matmul(eigvec[:,:eig_num], torch.diag(sort_eigval[:eig_num])), eigvec[:,:eig_num].T)
    return recon_cov

def positive_recons(matrix):
    eigval, eigvec = torch.linalg.eigh(matrix)

    eigval[eigval < 0] = 0
    pos_mat = torch.matmul(torch.matmul(eigvec, torch.diag(eigval)), eigvec.T)
    return pos_mat


def mean_confidence_interval(data, confidence=0.95):
    a = 100.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * t._ppf((1+confidence)/2., n-1)
    return m, h

class CosineSimilarity_miuns(nn.Module):
    def  __init__(self, in_features, out_features, scale_factor=5.0):
        super().__init__()
        self.scale_factor = scale_factor
        self.weight = Parameter(torch.Tensor(out_features, in_features).float())
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        self.cls = (torch.nn.Linear(in_features, out_features))
    def forward(self, feature, feature_base=None, training=True):

        cosine = F.linear(feature, F.normalize(self.weight))
        return cosine * self.scale_factor

if __name__ == '__main__':
    # ---- data loading
    dataset = 'miniImagenet'

    n_shot = 1
    n_ways = 5
    n_queries = 15
    n_runs = 600
    n_lsamples = n_ways * n_shot
    n_usamples = n_ways * n_queries
    n_samples = n_lsamples + n_usamples
    beta = 0.6
    import FSLTask
    cfg = {'shot': n_shot, 'ways': n_ways, 'queries': n_queries}
    FSLTask.loadDataSet(dataset)
    FSLTask.setRandomStates(cfg)
    ndatas = FSLTask.GenerateRunSet(end=n_runs, cfg=cfg)
    ndatas = ndatas.permute(0, 2, 1, 3).reshape(n_runs, n_samples, -1)
    labels = torch.arange(n_ways).view(1, 1, n_ways).expand(n_runs, n_shot + n_queries, 5).clone().view(n_runs,
                                                                                                        n_samples)
    # ---- Base class statistics
    all_features = []
    base_means = []
    base_cov = []

    if dataset == 'miniImagenet':
        base_features_path = "./%s/base_features.plk" % dataset
        with open(base_features_path, 'rb') as f:
            data = pickle.load(f)
            for key in data.keys():
                feature = np.array(data[key])
                # feature = np.power(feature[:,], beta)
                all_features.append(feature)
                mean = np.mean(feature, axis=0)
                cov = np.cov(feature.T)
                base_means.append(mean)
                base_cov.append(cov)

    all_features = np.vstack(all_features)

    base_cov = torch.Tensor(np.array(base_cov)).cuda()
    base_means = torch.Tensor(np.array(base_means)).cuda()
    # ---- classification for each task
    acc_list = []
    print('Start classification for %d tasks...'%(n_runs))

    save_path = r'./result_{}_{}shot'.format(dataset, str(n_shot))
    os.makedirs(save_path, exist_ok=True)

    all_features = torch.Tensor(all_features).cuda()

    for i in (range(n_runs)):
        run_num = i

        support_data = ndatas[i][:n_lsamples]
        support_label = labels[i][:n_lsamples]
        query_data = ndatas[i][n_lsamples:]
        query_label = labels[i][n_lsamples:]

        support_data = F.relu(support_data)
        support_data = torch.pow(support_data[:, ], beta)
        query_data = F.relu(query_data)
        query_data = torch.pow(query_data[:, ], beta)

        # support_data = F.normalize(support_data)
        support_data = torch.Tensor(support_data).cuda()
        support_label = torch.LongTensor(support_label).cuda()
        query_data = torch.Tensor(query_data)
        query_label = torch.LongTensor(query_label)


        # ---- distribution calibration and feature sampling
        sampled_data = []
        sampled_label = []
        gen_sampled = torch.tensor([1000 // 1], dtype=torch.int32).cuda()

        for ind in range(n_lsamples):
            cls_mean, cls_cov = distribution_calibration(support_data[ind], base_means, base_cov,
                                                         all_features, k=2, use_smp=False)
            smp_mean, smp_cov = distribution_calibration(support_data[ind], base_means, base_cov,
                                                         all_features, k=2, use_smp=True)

            recon_cls_cov = recons_cov(cls_cov, 110)
            recon_smp_cov = recons_cov(smp_cov, 160)

            cali_mean = 0.3 * cls_mean + 0.7 * smp_mean
            # cali_mean = 0.2 * cls_mean + 0.8 * smp_mean

            cali_cov =  1 * recon_cls_cov + 1 * recon_smp_cov + 0.4 + \
                        0.01 * (torch.diag(cls_cov).diag()) + 0.01 * (torch.diag(smp_cov).diag())

            distrib = torch.distributions.multivariate_normal.MultivariateNormal(cali_mean, covariance_matrix=cali_cov)

            gen_num = 0
            tmp_data = list()
            tmp_label = list()
            while gen_num < 500:
                gen_data = distrib.sample(gen_sampled)

                cos_sim = torch.cdist(support_data[ind][None, :], gen_data, p=2)
                cos_idx = torch.argwhere(((cos_sim)).squeeze() < 10)

                slt_num = cos_idx.shape[0]
                if slt_num > 0:
                    # select_gen_data = gen_data[cos_idx.squeeze()].reshape(-1,512)
                    select_gen_data = gen_data[cos_idx.squeeze()].reshape(-1, 640)
                    # select_gen_data = gen_data[cos_idx.squeeze()].reshape(-1, 768)
                    tmp_data.extend(select_gen_data)
                    tmp_label.extend([support_label[ind]] * slt_num)
                    gen_num += slt_num

            sampled_data.extend(tmp_data)
            sampled_label.extend(tmp_label)

        sampled_data = torch.vstack(sampled_data)
        sampled_label = torch.stack(sampled_label)

        #     # ---- train classifier
        #     X_aug = torch.cat((support_data, sampled_data), dim=0)
        #     Y_aug = torch.cat((support_label, sampled_label), dim=0)
        #     X_aug = F.normalize(X_aug, dim=1)
        #
        #     classifier = LogisticRegression(max_iter=1000).fit(X=X_aug.cpu().numpy(), y=Y_aug.cpu().numpy())
        #
        #     query_data = F.normalize(query_data, dim=1)
        #     predicts = classifier.predict(query_data.cpu().numpy())
        #     acc = np.mean(predicts == query_label.cpu().numpy())
        #     acc_list.append(acc)
        #
        #     f = open(os.path.join(save_path, '{}_{}'.format(i, acc)), 'w')
        #     f.close()
        #
        #     print('{}, acc: {}'.format(i, float(np.mean(acc_list))))
        # confi = str(round(1.96 * np.std(acc_list) / np.sqrt(len(acc_list))*100, 2))
        # print('%s %d way %d shot  ACC : %f, confi: %s'%(dataset,n_ways,n_shot,float(np.mean(acc_list)), confi))

        # ---- train  classification model
        model = CosineSimilarity_miuns(640, 5, scale_factor=5).cuda()

        # finetune_optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)
        finetune_optimizer = torch.optim.Adam(model.parameters(), 0.01, weight_decay=5e-4)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer=finetune_optimizer, step_size=150, gamma=0.9)
        loss_function = nn.CrossEntropyLoss().cuda()
        support_size = n_lsamples
        epoch_list = list()
        for epoch in (range(1000)):
            z_batch = torch.cat((sampled_data,  support_data), dim=0)
            y_batch = torch.cat((sampled_label, support_label), dim=0)


            scores = model(z_batch)
            loss = loss_function(scores, y_batch)
            tol_loss = loss

            finetune_optimizer.zero_grad()
            tol_loss.backward()
            finetune_optimizer.step()

        # query_data = F.normalize(query_data)
        output = model.eval()((query_data.cuda())).detach()
        with torch.no_grad():
            pred = output.argmax(dim=1)
            acc = (pred == query_label.cuda()).float().mean().cpu().numpy()
            epoch_list.append(acc)
        acc_list.append(acc)
        print('{}: acc{}'.format(run_num, np.mean(acc_list)))
    confi = str(round(1.96 * np.std(acc_list) / np.sqrt(len(acc_list)) * 100, 2))
    print('%s %d way %d shot  ACC : %f, Confi : %s' % (dataset, n_ways, n_shot, float(np.mean(acc_list)), confi))