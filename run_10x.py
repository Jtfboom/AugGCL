import argparse
import os
import csv
import torch
import random
import numpy as np
import gc
from train_visium import train_10x  # 假设这是你的训练函数


def run_with_params(opt, seed, rad_cutoff):
    """运行训练并返回 ARI 和 NMI 分数"""
    set_seed(seed)
    opt.seed = seed
    opt.rad_cutoff = rad_cutoff

    # 运行训练并获得 ARI 和 NMI 分数
    ARI, NMI = train_10x(opt)  # 假设 train_10x 是训练函数
    return ARI, NMI


def set_seed(seed):
    """设置随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_results_to_log(results, log_file):
    """将结果保存到日志文件"""
    with open(log_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(results)


# 主函数
if __name__ == "__main__":

    # 示例参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='10x', help='数据集类型')
    parser.add_argument('--lr', type=float, default=0.001)  # 学习率
    parser.add_argument('--root', type=str, default='./dataset/DLPFC/')
    parser.add_argument('--epochs', type=int, default=1200)
    parser.add_argument('--id', type=str, default='151507', help='样本 ID')  # 样本ID
    parser.add_argument('--seed', type=int, default=1314)
    parser.add_argument('--save_path', type=str, default='./checkpoint/10x_train/')
    parser.add_argument('--ncluster', type=int, default=7)
    parser.add_argument('--rad_cutoff', type=int, default=300)

    # 获取命令行参数
    opt = parser.parse_args()

    # 提取的样本数据
    sample_data = [
        ('151507', 262, 390),
        ('151508', 249, 400),
        ('151509', 269, 600),
        ('151510', 199, 300),
        ('151669', 808, 325),
        ('151670', 369, 200),
        ('151671', 296, 300),
        ('151672', 469, 150),
        ('151673', 403, 350),
        ('151674', 155, 375),
        ('151675', 485, 525),
        ('151676', 811, 375)
    ]

    # 创建日志文件并写入标题
    log_file = "DLPFC_ari_nmi_results.csv"
    if not os.path.exists(log_file):
        with open(log_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Sample ID', 'ARI', 'NMI'])

    # 对每个样本进行训练并保存结果
    for sample_id, seed, rad_cutoff in sample_data:
        print(f"Processing Sample ID: {sample_id}, Seed: {seed}, Rad Cutoff: {rad_cutoff}")
        # 更新 opt 参数中的样本 ID
        opt.id = sample_id
        # 如果样本 ID 在特定范围内，设置 ncluster 为 5
        if sample_id in ['151669', '151670', '151671', '151672']:
            opt.ncluster = 5
        else:
            opt.ncluster = 7

        # 运行训练并获得 ARI 和 NMI
        ari, nmi = run_with_params(opt, seed, rad_cutoff)

        # 保存结果到日志文件
        save_results_to_log([sample_id, ari, nmi], log_file)

    print("所有样本处理完成，结果已保存。")
