from torch_geometric.nn import TransformerConv, LayerNorm, GATConv, GCNConv
from torch_geometric.nn import GCNConv, BatchNorm,SAGEConv
# from torch_geometric.utils import shuffle_node, mask_feature
import torch.nn.functional as F
import torch
import math


class TransImg(torch.nn.Module):
    def __init__(self, hidden_dims, use_img_loss=False):
        super().__init__()
        [in_dim, img_dim, num_hidden, out_dim] = hidden_dims
        # [in_dim, emb_dim, img_dim, num_hidden, out_dim] = hidden_dims

        self.conv1 = GCNConv(in_dim, num_hidden)
        self.conv2 = GCNConv(num_hidden, out_dim)
        self.conv3 = GCNConv(out_dim, num_hidden)
        self.conv4 = GCNConv(num_hidden, in_dim)

        self.imgconv1 = GCNConv(img_dim, num_hidden)
        self.imgconv2 = GCNConv(num_hidden, out_dim)
        self.imgconv3 = GCNConv(out_dim, num_hidden)
        if use_img_loss:
            self.imgconv4 = GCNConv(num_hidden, img_dim)
        else:
            self.imgconv4 = GCNConv(num_hidden, in_dim)

        self.neck = GCNConv(out_dim * 2, out_dim)
        self.neck2 = GCNConv(out_dim, out_dim)
        self.c3 = GCNConv(out_dim, num_hidden)
        self.c4 = GCNConv(num_hidden, in_dim)

        # layernorm
        self.norm1 = LayerNorm(num_hidden)
        self.norm2 = LayerNorm(out_dim)
        # relu
        self.activate = F.elu



    def forward(self, features, img_feat, gene_edge, img_edge, gene_attr, img_attr):

        # 基因特征卷积
        h1 = self.activate(self.conv1(features, gene_edge, edge_weight=gene_attr))#, edge_weight=gene_attr
        h2 = self.conv2(h1, gene_edge, edge_weight=gene_attr)
        h3 = self.activate(self.conv3(h2, gene_edge))
        h4 = self.conv4(h3, gene_edge)

        # 图像特征卷积, edge_weight= img_attr
        img1 = self.activate(self.imgconv1(img_feat, img_edge, edge_weight= img_attr))#, edge_weight= img_attr
        img2 = self.imgconv2(img1, img_edge, edge_weight= img_attr)
        img3 = self.activate(self.imgconv3(img2, img_edge))
        img4 = self.imgconv4(img3, img_edge)

        concat = torch.cat([h2, img2], dim=1)
        combine = self.activate(self.neck(concat, gene_edge))
        c2 = self.neck2(combine, gene_edge)
        c3 = self.activate(self.c3(c2, gene_edge))
        c4 = self.c4(c3, gene_edge)

        return h2, img2, c2, h4, img4, c4

class TransGene(torch.nn.Module):
    def __init__(self, hidden_dims, use_img_loss=False):
        super().__init__()
        [in1_dim, in2_dim, num_hidden, out_dim] = hidden_dims
        # [in_dim, emb_dim, img_dim, num_hidden, out_dim] = hidden_dims

        self.conv1 = GCNConv(in1_dim, num_hidden)
        self.conv2 = GCNConv(num_hidden, out_dim)
        self.conv3 = GCNConv(out_dim, num_hidden)
        self.conv4 = GCNConv(num_hidden, in1_dim)

        self.imgconv1 = GCNConv(in2_dim, num_hidden)
        self.imgconv2 = GCNConv(num_hidden, out_dim)
        self.imgconv3 = GCNConv(out_dim, num_hidden)
        if use_img_loss:
            self.imgconv4 = GCNConv(num_hidden, in2_dim)
        else:
            self.imgconv4 = GCNConv(num_hidden, in2_dim)

        self.neck = GCNConv(out_dim * 2, out_dim)
        self.neck2 = GCNConv(out_dim, out_dim)
        self.c3 = GCNConv(out_dim, num_hidden)
        self.c4 = GCNConv(num_hidden, in1_dim)

        # layernorm
        self.norm1 = LayerNorm(num_hidden)
        self.norm2 = LayerNorm(out_dim)
        # relu
        self.activate = F.elu



    def forward(self, features_aug, features, gene_edge, img_edge, gene_attr, img_attr):

        # 基因特征卷积
        h1 = self.activate(self.conv1(features_aug, gene_edge, gene_attr))#, edge_weight=gene_attr
        h2 = self.conv2(h1, gene_edge, gene_attr)
        h3 = self.activate(self.conv3(h2, gene_edge))
        h4 = self.conv4(h3, gene_edge)

        # 图像特征卷积, edge_weight= img_attr
        img1 = self.activate(self.imgconv1(features, img_edge, img_attr))#, edge_weight= img_attr
        img2 = self.imgconv2(img1, img_edge, img_attr)
        img3 = self.activate(self.imgconv3(img2, img_edge))
        img4 = self.imgconv4(img3, img_edge)

        # 基因特征和图像特征的结合
        concat = torch.cat([h2, img2], dim=1)
        combine = self.activate(self.neck(concat, gene_edge))
        c2 = self.neck2(combine, gene_edge)
        c3 = self.activate(self.c3(c2 , gene_edge))
        c4 = self.c4(c3, gene_edge)

        return h2, img2, c2 , h4, img4, c4

