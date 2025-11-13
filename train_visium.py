import pandas as pd
import os
import scanpy as sc
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
import numpy as np
import cv2
import torchvision.transforms as transforms
from utils import Cal_Spatial_Net, Stats_Spatial_Net,  seed_everything
from train_AugGCL import train_img
import matplotlib.pyplot as plt
import ot

def refine_label(adata, radius=50, key='label'):
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values

    # calculate distance
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')

    n_cell = distance.shape[0]

    for i in range(n_cell):
        vec = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh + 1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)

    new_type = [str(i) for i in list(new_type)]
    return new_type

def train(opt,alpha1,alpha2):
    seed_everything(opt.seed)
    adata = sc.read(os.path.join(opt.root, opt.id, 'sampledata.h5ad'))
    adata.var_names_make_unique()
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000, check_values=False)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)


    labels_path = (os.path.join(opt.root, opt.id, "metadata.tsv"))
    labels = pd.read_table(labels_path, sep='\t')
    labels = labels["layer_guess_reordered"].copy()
    NA_labels = np.where(labels.isnull())
    labels = labels.drop(labels.index[NA_labels])
    ground = labels.copy()
    ground.replace('WM', '0', inplace=True)
    ground.replace('Layer1', '1', inplace=True)
    ground.replace('Layer2', '2', inplace=True)
    ground.replace('Layer3', '3', inplace=True)
    ground.replace('Layer4', '4', inplace=True)
    ground.replace('Layer5', '5', inplace=True)
    ground.replace('Layer6', '6', inplace=True)


    adata.obs['Ground Truth'] = labels
    adata.obs['ground'] = ground


    img = cv2.imread(os.path.join(opt.root,opt.id, 'spatial/full_image.tif'))
    transform = transforms.ToTensor()
    img = transform(img)

    patchs = []
    for coor in adata.obsm['spatial']:
        py, px = coor
        img_p = img[:, px-25:px+25, py-25:py+25].flatten()
        patchs.append(img_p)
    patchs = np.stack(patchs)
    df = pd.DataFrame(patchs, index=adata.obs.index)

    adata.obsm['imgs'] = df


    Cal_Spatial_Net(adata, rad_cutoff= opt.rad_cutoff,model='Radius')#150
    Stats_Spatial_Net(adata)


    sp = os.path.join(opt.save_path, opt.id)
    if not os.path.exists(sp):
        os.makedirs(sp)

    adata = train_img(adata, alpha1, alpha2,  hidden_dims=[512, 30],  n_epochs=opt.epochs,
                lr=opt.lr, random_seed=opt.seed, save_path=sp, ncluster=opt.ncluster, use_combine=0, use_img_loss=1)

    obs_df = adata.obs.dropna()
    ARI = adjusted_rand_score(obs_df['idx'], obs_df['Ground Truth'])
    NMI = normalized_mutual_info_score(obs_df['idx'], obs_df['Ground Truth'])
    print('ari is %.2f'%(ARI))

    result_path = './results/10x_final/'
    # plt.rcParams["figure.figsize"] = (3, 3)
    save_path = os.path.join(result_path, opt.id)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plt.rcParams["figure.figsize"] = (3, 3)
    sc.settings.figdir = save_path
    ax = sc.pl.spatial(adata, color=['idx'], title=['AugGCL (ARI=%.2f)' % (ARI)], show=False)
    plt.savefig(os.path.join(save_path, 'spatial.pdf'), bbox_inches='tight')

    sc.pp.neighbors(adata, n_neighbors=20, use_rep='pred')
    sc.tl.umap(adata)
    plt.rcParams["figure.figsize"] = (3, 3)
    sc.settings.figdir = save_path
    ax = sc.pl.umap(adata, color=['Ground Truth'], show=False, title='AugGCL', legend_loc='on data')
    plt.savefig(os.path.join(save_path, 'umap_final.pdf'), bbox_inches='tight')

    # Write the results to a file
    df = pd.DataFrame(adata.layers['recon'], index=adata.obs.index, columns=adata.var.index)
    df2 = pd.DataFrame(adata.obs['idx'], index=adata.obs.index)
    df2['Ground Truth'] = adata.obs['Ground Truth']
    X_umap = adata.obsm['X_umap']
    spatial_xy = adata.obsm['spatial']

    df2['umap_x'] = X_umap[:, 0]
    df2['umap_y'] = X_umap[:, 1]
    df2['spa_x'] = spatial_xy[:, 0]
    df2['spa_y'] = spatial_xy[:, 0]

    # Save adata as a new results file
    adata.obsm['imgs'] = adata.obsm['imgs'].to_numpy()
    adata.write('%s/%s_results.h5ad' % (opt.save_path, opt.id))
    print('Results saved to: %s/%s_results.h5ad' % (opt.save_path, opt.id))

    return ARI,NMI


def train_10x(opt):
        # 确保保存路径存在
        if not os.path.exists(os.path.join(opt.save_path, opt.id)):
            os.makedirs(os.path.join(opt.save_path, opt.id))

        # 运行训练并获取 ARI 值
        ARI, NMI = train(opt, alpha1=1.2, alpha2=0.8)
        print(f'当前 ARI: {ARI:.2f}')

        return ARI,NMI