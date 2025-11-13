import copy
import os
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import issparse
import sklearn.neighbors
import random
from torch_geometric.data import Data
import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = False
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances
from scipy.sparse import csr_matrix
import torch
import scipy.sparse as sp


def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='SpaCAE', random_seed=100):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """

    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    from rpy2.robjects import r

    # 设置R的locale为UTF-8
    r('Sys.setlocale("LC_ALL", "en_US.UTF-8")')
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def  Transfer_img_Data(adata, alpha1=1.1, alpha2=0.8):
    G_df = adata.uns['Spatial_Net'].copy()
    cells = np.array(adata.obs_names)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)

    e0 = G_df['Cell1'].to_numpy()
    e1 = G_df['Cell2'].to_numpy()
    edgeList = np.array((e0, e1))

    # 生成增强矩阵
    exmatrix_ori, exmatrix_aug, imgs_aug, gene_adj, gene_adj_attr, img_adj, img_adj_attr = NIA_aug(adata, alpha1, alpha2)

    if type(adata.X) == np.ndarray:
        if 'X_train' in adata.obs.keys():
            X_train_idx = (adata.obs['X_train'].to_numpy() == 1)
            X_test_idx = (adata.obs['X_train'].to_numpy() == 0)

            data = Data(edge_index=gene_adj,
                x=torch.FloatTensor(adata.X),
                x_aug=torch.FloatTensor(exmatrix_aug.todense()),
                edge_attr=torch.FloatTensor(gene_adj_attr),  # 添加边特征
                train_mask=list(X_train_idx),
                val_mask=list(X_test_idx))

            img = Data(edge_index=img_adj,
                x=torch.FloatTensor(adata.obsm['imgs']),
                aug=torch.FloatTensor(imgs_aug.todense()),
                edge_attr=torch.FloatTensor(img_adj_attr),  # 添加图像边特征
                train_mask=list(X_train_idx),
                val_mask=list(X_test_idx))
        else:
            data = Data(edge_index=gene_adj,
                x=torch.FloatTensor(adata.X),
                x_aug=torch.FloatTensor(exmatrix_aug.todense()),
                edge_attr=torch.FloatTensor(gene_adj_attr))  # 添加边特征

            img = Data(edge_index=img_adj,
                x=torch.FloatTensor(adata.obsm['imgs'].to_numpy()),
                aug=torch.FloatTensor(imgs_aug.todense()),
                edge_attr=torch.FloatTensor(img_adj_attr))  # 添加图像边特征
    else:
        if 'X_train' in adata.obs.keys():
            X_train_idx = (adata.obs['X_train'].to_numpy() == 1)
            X_test_idx = (adata.obs['X_train'].to_numpy() == 0)

            data = Data(edge_index=gene_adj,
                x=torch.FloatTensor(adata.X.todense()),
                x_aug=torch.FloatTensor(exmatrix_aug.todense()),
                edge_attr=torch.FloatTensor(gene_adj_attr),  # 添加边特征
                train_mask=list(X_train_idx),
                val_mask=list(X_test_idx))

            img = Data(edge_index=img_adj,
                x=torch.FloatTensor(adata.obsm['imgs'].to_numpy()),
                aug=torch.FloatTensor(imgs_aug.todense()),
                edge_attr=torch.FloatTensor(img_adj_attr),  # 添加图像边特征
                train_mask=list(X_train_idx),
                val_mask=list(X_test_idx))
        else:
            data = Data(edge_index=gene_adj,
                x=torch.FloatTensor(adata.X.todense()),
                x_aug=torch.FloatTensor(exmatrix_aug.todense()),
                edge_attr=torch.FloatTensor(gene_adj_attr))  # 添加边特征

            img = Data(edge_index=img_adj,
                x = torch.FloatTensor(adata.obsm['imgs'].to_numpy()),
                aug=torch.FloatTensor(imgs_aug.todense()),
                edge_attr=torch.FloatTensor(img_adj_attr))  # 添加图像边特征


    return data, img

def Transfer_Data(adata, alpha1=1.1):
    G_df = adata.uns['Spatial_Net'].copy()
    cells = np.array(adata.obs_names)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)

    e0 = G_df['Cell1'].to_numpy()
    e1 = G_df['Cell2'].to_numpy()
    edgeList = np.array((e0, e1))

    # 生成增强矩阵
    exmatrix_ori, exmatrix_aug, gene_adj, gene_adj_attr = NIA_noimg_aug(adata, alpha1)


    if type(adata.X) == np.ndarray:
        if 'X_train' in adata.obs.keys():
            X_train_idx = (adata.obs['X_train'].to_numpy() == 1)
            X_test_idx = (adata.obs['X_train'].to_numpy() == 0)

            data = Data(edge_index=gene_adj,
                x=torch.FloatTensor(adata.X),
                x_aug=torch.FloatTensor(exmatrix_aug.todense()),
                edge_attr=torch.FloatTensor(gene_adj_attr),  # 添加边特征
                train_mask=list(X_train_idx),
                val_mask=list(X_test_idx))

        else:
            data = Data(edge_index=gene_adj,
                x=torch.FloatTensor(adata.X),
                x_aug=torch.FloatTensor(exmatrix_aug.todense()),
                edge_attr=torch.FloatTensor(gene_adj_attr))  # 添加边特征

    else:
        if 'X_train' in adata.obs.keys():
            X_train_idx = (adata.obs['X_train'].to_numpy() == 1)
            X_test_idx = (adata.obs['X_train'].to_numpy() == 0)

            data = Data(edge_index=gene_adj,
                x=torch.FloatTensor(adata.X.todense()),
                x_aug=torch.FloatTensor(exmatrix_aug.todense()),
                edge_attr=torch.FloatTensor(gene_adj_attr),  # 添加边特征
                train_mask=list(X_train_idx),
                val_mask=list(X_test_idx))

        else:
            data = Data(edge_index=gene_adj,
                x=torch.FloatTensor(adata.X.todense()),
                x_aug=torch.FloatTensor(exmatrix_aug.todense()),
                edge_attr=torch.FloatTensor(gene_adj_attr))  # 添加边特征



    return data


def Cal_Spatial_Net(adata, rad_cutoff=None, k_cutoff=None, model='Radius', verbose=True, use_global=False):
    if verbose:
        print('------Calculating spatial graph...')
    if use_global:
        coor = pd.DataFrame(adata.obsm['spatial_global'])
    else:
        coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']

    if model == 'Radius':
        nbrs = sklearn.neighbors.NearestNeighbors(radius=rad_cutoff).fit(coor)
        distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it] * indices[it].shape[0], indices[it], distances[it])))


    if model == 'KNN':
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff + 1).fit(coor)
        distances, indices = nbrs.kneighbors(coor)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it] * indices.shape[1], indices[it, :], distances[it, :])))

    KNN_df = pd.concat(KNN_list)
    # print(KNN_df.shape)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

    Spatial_Net = KNN_df.copy()
    Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance'] > 0,]
    adata.uns['Spatial_Net_ori'] = copy.deepcopy(Spatial_Net)
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
    if verbose:
        print('The graph contains %d edges, %d cells.' % (Spatial_Net.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' % (Spatial_Net.shape[0] / adata.n_obs))

    adata.uns['Spatial_Net'] = Spatial_Net



def Stats_Spatial_Net(adata):
    import matplotlib.pyplot as plt
    Num_edge = adata.uns['Spatial_Net']['Cell1'].shape[0]
    Mean_edge = Num_edge / adata.shape[0]
    plot_df = pd.value_counts(pd.value_counts(adata.uns['Spatial_Net']['Cell1']))
    plot_df = plot_df / adata.shape[0]
    fig, ax = plt.subplots(figsize=[3, 2])
    plt.ylabel('Percentage')
    plt.xlabel('')
    plt.title('Number of Neighbors (Mean=%.2f)' % Mean_edge)
    ax.bar(plot_df.index, plot_df)
    plt.close('all')



def NIA_noimg_aug(adata, alpha1):  # 添加n_components和n_neighbors参数
    # 提取高度可变基因
    if 'highly_variable' in adata.var.columns:
        hvg = list(adata.var['highly_variable'][adata.var['highly_variable'].values].index)
    else:
        hvg = adata.var_names.tolist()

    # 提取原始表达矩阵
    exmatrix_ori = adata.to_df()[hvg].to_numpy()


    # 计算空间邻接矩阵
    nbrs = adata.uns['Spatial_Net_ori']

    df = pd.DataFrame(nbrs)
    num_nodes = exmatrix_ori.shape[0]

    adj_matrix = np.zeros((int(num_nodes), int(num_nodes)))

    for idx, row in df.iterrows():
        cell1 = int(row['Cell1'])
        cell2 = int(row['Cell2'])
        adj_matrix[cell1, cell2] = 1

    adj_matrix = adj_matrix.toarray() if isinstance(adj_matrix, csr_matrix) else adj_matrix

    # 计算基因特征余弦距离矩阵
    gene_dists = np.exp(2 - cosine_distances(exmatrix_ori)) - 1

    # 计算空间转移概率矩阵
    gene_conns = adj_matrix.T * gene_dists

    # 归一化
    gene_sum = np.sum(gene_conns, axis=0, keepdims=True)
    gene_stg = np.where(gene_sum != 0, gene_conns / (gene_sum + 1e-10), 0)

    # 生成增强后的表达矩阵
    X = adata.X.toarray() if issparse(adata.X) else adata.X
    X_rec =(alpha1) * np.matmul(gene_stg, X) + X

    exmatrix_aug = csr_matrix(X_rec)


    adj_sparse = sp.coo_matrix(gene_stg)
    gene_adj = torch.tensor(np.vstack((adj_sparse.row, adj_sparse.col)), dtype=torch.long)
    gene_adj_attr = torch.tensor(adj_sparse.data, dtype=torch.float)


    return exmatrix_ori, exmatrix_aug, gene_adj, gene_adj_attr


def NIA_aug(adata, alpha1, alpha2):
    # 提取高度可变基因
    if 'highly_variable' in adata.var.columns:
        hvg = list(adata.var['highly_variable'][adata.var['highly_variable'].values].index)
    else:
        hvg = adata.var_names.tolist()

    # 提取原始表达矩阵
    exmatrix_ori = adata.to_df()[hvg].to_numpy()

    imgs_ori = adata.obsm['imgs'].copy()

    # 计算空间邻接矩阵
    nbrs = adata.uns['Spatial_Net_ori']
    df = pd.DataFrame(nbrs)
    num_nodes = exmatrix_ori.shape[0]
    adj_matrix = np.zeros((int(num_nodes), int(num_nodes)))  # 初始化邻接矩阵


    for idx, row in df.iterrows():
        cell1 = int(row['Cell1'])
        cell2 = int(row['Cell2'])
        adj_matrix[cell1, cell2] = 1  # 直接连接的细胞设为1

    # 计算基因特征余弦距离矩阵
    gene_dists = np.exp(2 - cosine_distances(exmatrix_ori)) - 1
    img_dists = np.exp(2 - cosine_distances(imgs_ori)) - 1

    # 计算空间转移概率矩阵
    gene_conns = adj_matrix.T * gene_dists
    img_conns = adj_matrix.T * img_dists

    # 归一化
    gene_sum = np.sum(gene_conns, axis=0, keepdims=True)
    gene_stg = np.where(gene_sum != 0, gene_conns / (gene_sum + 1e-10), 0)

    img_sum = np.sum(img_conns, axis=0, keepdims=True)
    img_stg = np.where(img_sum != 0, img_conns / (img_sum + 1e-10), 0)

    # 生成增强后的表达矩阵
    X = adata.X.toarray() if issparse(adata.X) else adata.X
    X_rec = (alpha1) * np.matmul(gene_stg, X) + X  # 基因增强
    exmatrix_aug = csr_matrix(X_rec)

    # 确保 imgs 是一个 NumPy 数组
    imgs = imgs_ori.toarray() if issparse(imgs_ori) else np.array(imgs_ori)
    # 进行矩阵乘法
    imgs_aug1 = (alpha2) * np.matmul(img_stg, imgs)

    # 确保 imgs_aug1 和 imgs 的形状相同
    assert imgs_aug1.shape == imgs.shape, "Shapes of imgs_aug1 and imgs do not match for addition."

    # 进行加法操作
    imgs_aug = imgs_aug1 + imgs

    # 将结果转换为稀疏矩阵（如果需要）
    imgs_aug_sparse = csr_matrix(imgs_aug)

    # 转换为 PyTorch 张量
    adj_sparse = sp.coo_matrix(gene_stg)
    gene_adj = torch.tensor(np.vstack((adj_sparse.row, adj_sparse.col)), dtype=torch.long)

    # 计算边的数量
    num_edges = gene_adj.shape[1]  # 获取列数，即边的数量

    gene_adj_attr = torch.tensor(adj_sparse.data, dtype=torch.float)

    adj_sparse1 = sp.coo_matrix(img_stg)
    img_adj = torch.tensor(np.vstack((adj_sparse1.row, adj_sparse1.col)), dtype=torch.long)
    img_adj_attr = torch.tensor(adj_sparse1.data, dtype=torch.float)

    # 判断 gene_adj 和 adj_matrix 是否相同
    def check_equal(gene_adj, adj_matrix):
        # 将gene_adj转换为稀疏矩阵
        gene_adj_sparse = sp.coo_matrix((np.ones(gene_adj.shape[1]), (gene_adj[0], gene_adj[1])),
                                        shape=adj_matrix.shape)

        # 比较两个稀疏矩阵是否相同
        return (gene_adj_sparse.row == adj_matrix.row).all() and (gene_adj_sparse.col == adj_matrix.col).all() and \
               (gene_adj_sparse.data == adj_matrix.data).all()



    return exmatrix_ori, exmatrix_aug, imgs_aug_sparse, gene_adj, gene_adj_attr, img_adj, img_adj_attr


