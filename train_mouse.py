import os
import scanpy as sc
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
import numpy as np
from utils import Cal_Spatial_Net, Stats_Spatial_Net,  seed_everything
from train_AugGCL import train_Gene
import matplotlib.pyplot as plt


def train(opt, alpha1=1.2):
    seed_everything(opt.seed)
    adata = sc.read(os.path.join(opt.root, 'E1S1.h5ad'))
    adata.var_names_make_unique()
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000, check_values=False)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    if 'annotation' in adata.obs.columns:
        labels = adata.obs['annotation'].copy()
    else:
        raise ValueError("标签（annotation）列在 adata.obs 中不存在！")

        # 移除缺失值（NA）
    NA_labels = np.where(labels.isnull())
    labels = labels.drop(labels.index[NA_labels])

    # 将标签值映射为您需要的数字标签
    ground = labels.copy()

    ground.replace('AGM', '0', inplace=True)
    ground.replace('Brain', '1', inplace=True)
    ground.replace('Branchial arch', '2', inplace=True)
    ground.replace('Cavity', '3', inplace=True)
    ground.replace('Connective tissue', '4', inplace=True)
    ground.replace('Dermomyotome', '5', inplace=True)
    ground.replace('Heart', '6', inplace=True)
    ground.replace('Liver', '7', inplace=True)
    ground.replace('Mesenchyme', '8', inplace=True)
    ground.replace('Neural crest', '9', inplace=True)
    ground.replace('Notochord', '10', inplace=True)
    ground.replace('Sclerotome', '11', inplace=True)


    # 将处理后的标签加入 adata.obs
    adata.obs['Ground Truth'] = labels
    adata.obs['ground'] = ground

    Cal_Spatial_Net(adata, rad_cutoff=opt.rad_cutoff)
    Stats_Spatial_Net(adata)

    sp = os.path.join(opt.save_path)
    if not os.path.exists(sp):
        os.makedirs(sp)

    adata = train_Gene(adata, alpha1, hidden_dims=[512, 30], n_epochs=opt.epochs,
                            lr=opt.lr, random_seed=opt.seed, save_path=sp, ncluster=opt.ncluster,
                            use_combine=0, use_img_loss=0)


    obs_df = adata.obs.dropna()

    ARI = adjusted_rand_score(obs_df['idx'], obs_df['Ground Truth'])
    NMI = normalized_mutual_info_score(obs_df['idx'], obs_df['Ground Truth'])

    result_path = './results/mouse_final/'

    save_path = os.path.join(result_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    #
    plt.rcParams["figure.figsize"] = (3, 3)
    sc.settings.figdir = save_path

    ax = sc.pl.spatial(adata, color=['annotation'], title=['Ground Truth'], spot_size=1, show=False)
    plt.savefig(os.path.join(save_path, f'gt_spatial.pdf'), bbox_inches='tight')

    ax = sc.pl.spatial(adata, color=['idx'], title=[f'AugGCL (ARI={ARI:.2f})'], spot_size=1, show=False)
    plt.savefig(os.path.join(save_path, f'spatial.pdf'), bbox_inches='tight')

    adata.write('%s_results.h5ad' % (opt.save_path))
    print('write to %s_results.h5ad' % (opt.save_path))

    return ARI, NMI




def train_mouse(opt):
    ARI, NMI = train(opt)
    print(f'当前 ARI: {ARI:.2f}')
    return ARI, NMI
