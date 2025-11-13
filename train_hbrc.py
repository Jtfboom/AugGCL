
import pandas as pd
import os
import scanpy as sc
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
import numpy as np
import cv2
import torchvision.transforms as transforms

from utils import Cal_Spatial_Net, Stats_Spatial_Net, mclust_R, seed_everything
from train_AugGCL import train_img
import matplotlib.pyplot as plt


def train(opt, alpha1, alpha2):
    seed_everything(opt.seed)
    adata = sc.read(os.path.join(opt.root, 'sampledata.h5ad'))
    adata.var_names_make_unique()
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000, check_values=False)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)


    labels_path = (os.path.join(opt.root, "metadata.tsv"))
    labels = pd.read_table(labels_path, sep='\t')
    labels.set_index('ID', inplace=True)
    labels = labels["ground_truth"].copy()




    NA_labels = np.where(labels.isnull())
    labels = labels.drop(labels.index[NA_labels])
    ground = labels.copy()

    ground.replace('DCIS/LCIS_1', '0', inplace=True)
    ground.replace('DCIS/LCIS_2', '1', inplace=True)
    ground.replace('DCIS/LCIS_3', '2', inplace=True)
    ground.replace('DCIS/LCIS_4', '3', inplace=True)
    ground.replace('Healthy_1', '4', inplace=True)
    ground.replace('Healthy_2', '5', inplace=True)
    ground.replace('IDC_1', '6', inplace=True)
    ground.replace('IDC_2', '7', inplace=True)
    ground.replace('IDC_3', '8', inplace=True)
    ground.replace('IDC_4', '9', inplace=True)
    ground.replace('IDC_5', '10', inplace=True)
    ground.replace('IDC_6', '11', inplace=True)
    ground.replace('IDC_7', '12', inplace=True)
    ground.replace('IDC_8', '13', inplace=True)
    ground.replace('Tumor_edge_1', '14', inplace=True)
    ground.replace('Tumor_edge_2', '15', inplace=True)
    ground.replace('Tumor_edge_3', '16', inplace=True)
    ground.replace('Tumor_edge_4', '17', inplace=True)
    ground.replace('Tumor_edge_5', '18', inplace=True)
    ground.replace('Tumor_edge_6', '19', inplace=True)

    adata.obs['Ground Truth'] = labels
    adata.obs['ground'] = ground


    img = cv2.imread(os.path.join(opt.root, 'spatial/full_image.tif'))
    transform = transforms.ToTensor()
    img = transform(img)

    patchs = []
    for coor in adata.obsm['spatial']:
        py, px = coor
        img_p = img[:, px - 25:px + 25, py - 25:py + 25].flatten()
        patchs.append(img_p)
    patchs = np.stack(patchs)
    df = pd.DataFrame(patchs, index=adata.obs.index)

    adata.obsm['imgs'] = df


    Cal_Spatial_Net(adata, rad_cutoff = opt.rad_cutoff)
    Stats_Spatial_Net(adata)
    # generate_exmatrix_aug(adata, alpha=1.1)

    sp = os.path.join(opt.save_path)
    if not os.path.exists(sp):
        os.makedirs(sp)


    adata = train_img(adata, alpha1, alpha2, hidden_dims=[512, 64], n_epochs=opt.epochs,
                            lr=opt.lr, random_seed=opt.seed, save_path=sp, ncluster=opt.ncluster,
                            use_combine=0, use_img_loss=1)


    obs_df = adata.obs.dropna()
    #  = adjusted_rand_score(obs_df['mclust'], obs_df['Ground Truth'])
    ARI = adjusted_rand_score(obs_df['idx'], obs_df['Ground Truth'])
    NMI = normalized_mutual_info_score(obs_df['idx'], obs_df['Ground Truth'])

    result_path = './results/hbrc_final/'
        # plt.rcParams["figure.figsize"] = (3, 3)
    save_path = os.path.join(result_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plt.rcParams["figure.figsize"] = (3, 3)
    sc.settings.figdir = save_path
    ax = sc.pl.spatial(adata, color=['Ground Truth'], title=['Ground Truth'], show=False)
    plt.savefig(os.path.join(save_path, 'gt_spatial.pdf'), bbox_inches='tight')


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

    # write df
    df = pd.DataFrame(adata.layers['recon'], index=adata.obs.index, columns=adata.var.index)

    df2 = pd.DataFrame(adata.obs['idx'], index=adata.obs.index)
    df2['Ground Truth'] = adata.obs['Ground Truth']
    X_umap = adata.obsm['X_umap']
    spatial_xy = adata.obsm['spatial']

    df2['umap_x'] = X_umap[:, 0]
    df2['umap_y'] = X_umap[:, 1]
    df2['spa_x'] = spatial_xy[:, 0]
    df2['spa_y'] = spatial_xy[:, 0]


    adata.obsm['imgs'] = adata.obsm['imgs'].to_numpy()
    adata.write('%s_results.h5ad' % (opt.save_path))
    print('write to %s_results.h5ad' % (opt.save_path))

    return ARI, NMI



def train_hbrc(opt):

    ARI, NMI= train(opt, alpha1=1.2, alpha2=0.8)

    return ARI, NMI
#