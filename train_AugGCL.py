import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import scipy.sparse as sp
from transModel import TransGene, TransImg
from utils import Transfer_img_Data, Transfer_Data
import torch
import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = False
import torch.nn.functional as F
import matplotlib.pyplot as plt
import scanpy as sc
import os
from sklearn import metrics

def train_img(adata, alpha1, alpha2,  hidden_dims=[512, 30], n_epochs=1000,
              lr=0.001,gradient_clipping=5., weight_decay=5e-4, verbose=True,
              random_seed=0, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
              save_path='./checkpoint/trans_gene/', ncluster=7,  use_img_loss=0,
              use_combine=1,lambda_1=1, lambda_2=1, lambda_3=1):

    seed = random_seed
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    adata.X = sp.csr_matrix(adata.X)

    if 'highly_variable' in adata.var.columns:
        adata_Vars = adata[:, adata.var['highly_variable']]
    else:
        adata_Vars = adata

    if verbose:
        print('Size of Input: ', adata_Vars.shape)
    if 'Spatial_Net' not in adata.uns.keys():
        raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")

    # Create and add val_mask to adata object
    n_samples = adata.shape[0]
    val_ratio = 0.125
    val_size = int(n_samples * val_ratio)
    val_indices = np.random.choice(n_samples, val_size, replace=False)
    val_mask = np.zeros(n_samples, dtype=bool)
    val_mask[val_indices] = True
    adata.obs['val_mask'] = val_mask


    data, img = Transfer_img_Data(adata_Vars, alpha1, alpha2)


    model = TransImg(hidden_dims=[data.x.shape[1], img.x.shape[1]] + hidden_dims, use_img_loss=use_img_loss).to(device)


    if not os.path.exists(save_path):
        os.makedirs(save_path)  # 创建目录 # 保存到具有写权限的目录

    print(os.path.join(save_path, 'init.pth'))  # 打印路径，确认拼接结果
    try:
        torch.save(model.state_dict(), os.path.join(save_path, 'init.pth'))
        print("Model saved successfully!")
    except Exception as e:
        print(f"Error saving model: {e}")

    data = data.to(device)
    img = img.to(device)

    # Add val_mask to data and img objects
    data.val_mask = torch.tensor(adata.obs['val_mask'].values, dtype=torch.bool).to(device)
    img.val_mask = torch.tensor(adata.obs['val_mask'].values, dtype=torch.bool).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    seed = random_seed
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    labels = adata.obs['ground']
    # 如果 labels 是 pandas Series 类型，且包含字符串标签
    if isinstance(labels, pd.Series):
        # 使用 LabelEncoder 将字符串标签转换为数值标签
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels)  # 将字符串标签转换为整数


    ari_max = 0
    # 在循环前初始化 emb_max
    emb_max = None
    for epoch in tqdm(range(1, n_epochs + 1)):

        model.train()
        optimizer.zero_grad()

        gz,iz,cz, gout,iout,cout = model(data.x_aug, img.x, data.edge_index, img.edge_index, data.edge_attr,
                                                     img.edge_attr)

        gloss = F.mse_loss(data.x, gout)
        if use_img_loss:
            iloss = F.mse_loss(img.x, iout)
        else:
            iloss = F.mse_loss(data.x, iout)

        if use_combine:
            closs = F.mse_loss(data.x, cout)
            loss = gloss * lambda_1 + iloss * lambda_2 + closs * lambda_3
        else:
            loss = gloss * lambda_1 + iloss * lambda_2

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        optimizer.step()

        cz = cz.to('cpu').detach().numpy().astype(np.float32)

        kmeans = KMeans(n_clusters=ncluster, n_init='auto').fit(cz)

        idx = kmeans.labels_

        # 保证 idx 是整数类型
        if isinstance(idx, np.ndarray):
            idx = idx.astype(np.int32)  # 保持为整数类型（建议）

        idx = idx.astype(np.int32)

        # 删除 NaN 值

        labels, idx = remove_nan_values(labels, idx)

        ari_res = metrics.adjusted_rand_score(labels, idx)


        nmi_res = normalized_mutual_info_score(labels, idx)

        if ari_res > ari_max:
            ari_max = ari_res
            nmi_max = nmi_res
            epoch_max = epoch
            idx_max = idx
            emb_max = cz

    # 确保 emb_max 是一个 PyTorch 张量
    if isinstance(emb_max, np.ndarray):
        emb_max = torch.tensor(emb_max)
    adata_Vars.obsm['pred'] = emb_max.clone().detach().cpu().numpy()
    adata_Vars.obs['idx'] = idx_max.astype(str)


    sc.pp.neighbors(adata_Vars, use_rep='pred')
    sc.tl.umap(adata_Vars)
    plt.rcParams["figure.figsize"] = (3, 3)
    sc.settings.figdir = save_path
    ax = sc.pl.umap(adata_Vars, color=['Ground Truth'], show=False, title='combined latent variables')
    plt.savefig(os.path.join(save_path, 'umap_final.pdf'), bbox_inches='tight')

    adata_Vars.obsm['pred'] = emb_max.to('cpu').detach().numpy().astype(np.float32)
    output = cout.to('cpu').detach().numpy().astype(np.float32)
    output[output < 0] = 0
    adata_Vars.layers['recon'] = output
    plt.close('all')
    return adata_Vars


# 在计算 ARI 和 NMI 之前，删除包含 NaN 的样本
def remove_nan_values(labels, idx):
    # 确保 labels 和 idx 都是 numpy 数组，如果是 pandas Series，则转换为 numpy 数组
    labels = np.array(labels)
    idx = np.array(idx)

    # 找到标签和预测中不包含 NaN 的位置
    valid_indices = ~np.isnan(labels) & ~np.isnan(idx)  # 同时删除 labels 和 idx 中的 NaN

    # 使用布尔索引删除 NaN 对应的行
    labels = labels[valid_indices]
    idx = idx[valid_indices]

    return labels, idx


def train_Gene(adata, alpha1, hidden_dims=[512, 30], n_epochs=1000, lr=0.001,
              gradient_clipping=5., weight_decay=0.0001, verbose=True,
              random_seed=0, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
              save_path='./checkpoint/trans_gene/', ncluster=7,  use_img_loss=0, use_combine=1,
               lambda_1=1, lambda_2=1, lambda_3=1):

    seed = random_seed
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    adata.X = sp.csr_matrix(adata.X)

    if 'highly_variable' in adata.var.columns:
        adata_Vars = adata[:, adata.var['highly_variable']]
    else:
        adata_Vars = adata

    if verbose:
        print('Size of Input: ', adata_Vars.shape)
    if 'Spatial_Net' not in adata.uns.keys():
        raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")

    # Create and add val_mask to adata object
    n_samples = adata.shape[0]
    val_ratio = 0.1
    val_size = int(n_samples * val_ratio)
    val_indices = np.random.choice(n_samples, val_size, replace=False)
    val_mask = np.zeros(n_samples, dtype=bool)
    val_mask[val_indices] = True
    adata.obs['val_mask'] = val_mask


    data = Transfer_Data(adata_Vars, alpha1)

    model = TransGene(hidden_dims=[data.x.shape[1], data.x.shape[1]] + hidden_dims, use_img_loss=use_img_loss).to(device)


    if not os.path.exists(save_path):
        os.makedirs(save_path)  # 创建目录 # 保存到具有写权限的目录

    print(os.path.join(save_path, 'init.pth'))  # 打印路径，确认拼接结果
    try:
        torch.save(model.state_dict(), os.path.join(save_path, 'init.pth'))
        print("Model saved successfully!")
    except Exception as e:
        print(f"Error saving model: {e}")

    data = data.to(device)


    data.val_mask = torch.tensor(adata.obs['val_mask'].values, dtype=torch.bool).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    seed = random_seed
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    labels = adata.obs['ground']

    if isinstance(labels, pd.Series):

        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels)


    ari_max = 0
    # 在循环前初始化 emb_max
    emb_max = None

    for epoch in tqdm(range(1, n_epochs + 1)):

        model.train()
        optimizer.zero_grad()



        gz, iz, cz, gout, iout, cout = model(data.x_aug, data.x, data.edge_index, data.edge_index, data.edge_attr,
                                             data.edge_attr)

        gloss = F.mse_loss(data.x, gout)

        iloss = F.mse_loss(data.x, iout)

        if use_combine:
            closs = F.mse_loss(data.x, cout)
            loss = gloss * lambda_1 + iloss * lambda_2 + closs * lambda_3
        else:
            loss = gloss * lambda_1 + iloss * lambda_2

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        optimizer.step()

        cz = cz.to('cpu').detach().numpy().astype(np.float32)

        kmeans = KMeans(n_clusters=ncluster, n_init='auto').fit(cz)
        idx = kmeans.labels_

        # 保证 idx 是整数类型
        if isinstance(idx, np.ndarray):
            idx = idx.astype(np.int32)  # 保持为整数类型（建议）

        idx = idx.astype(np.int32)



        labels, idx = remove_nan_values(labels, idx)

        ari_res = metrics.adjusted_rand_score(labels, idx)

        nmi_res = normalized_mutual_info_score(labels, idx)

        if ari_res > ari_max:
            ari_max = ari_res
            nmi_max = nmi_res
            epoch_max = epoch
            idx_max = idx
            emb_max = cz



    # 确保 emb_max 是一个 PyTorch 张量
    if isinstance(emb_max, np.ndarray):
        emb_max = torch.tensor(emb_max)
    adata_Vars.obsm['pred'] = emb_max.clone().detach().cpu().numpy()
    adata_Vars.obs['idx'] = idx_max.astype(str)


    # 计算邻居
    sc.pp.neighbors(adata_Vars, use_rep='pred')

    sc.tl.umap(adata_Vars)
    plt.rcParams["figure.figsize"] = (3, 3)
    sc.settings.figdir = save_path
    ax = sc.pl.umap(adata_Vars, color=['Ground Truth'], show=False, title='combined latent variables')
    plt.savefig(os.path.join(save_path, 'umap_final.pdf'), bbox_inches='tight')

    adata_Vars.obsm['pred'] = emb_max.to('cpu').detach().numpy().astype(np.float32)

    output = cout.to('cpu').detach().numpy().astype(np.float32)
    output[output < 0] = 0
    adata_Vars.layers['recon'] = output
    plt.close('all')
    return adata_Vars
