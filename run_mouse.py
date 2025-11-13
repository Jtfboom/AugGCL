import argparse
from train_mouse import train_mouse  # 假设这是你的训练函数


# 示例用法
if __name__ == "__main__":
    # 示例参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mouse', help='数据集类型')
    parser.add_argument('--lr', type=float, default=0.001)  # 学习率
    parser.add_argument('--root', type=str, default='./dataset/mouse/')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--id', type=str, default='151507', help='样本 ID')  # 样本ID
    parser.add_argument('--seed', type=int, default=853)
    parser.add_argument('--save_path', type=str, default='./checkpoint/mouse_train/')
    parser.add_argument('--ncluster', type=int, default=12)
    parser.add_argument('--rad_cutoff', type=int, default=2)

    # 获取命令行参数
    opt = parser.parse_args()

    ari, nmi = train_mouse(opt)
    # 输出最终的最佳种子和 ARI
    print(f" 最佳 ARI: {ari}, 最佳 NMI: {nmi}")

