import argparse

def get_config():
    parser = argparse.ArgumentParser()
    # 实验设置
    parser.add_argument('--exp_name', type=str, default='baseline', help='实验名称')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--step_size', type=int, default=7)
    parser.add_argument('--gamma', type=float, default=0.1)
    
    # 模型参数
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18'])
    
    # 路径参数
    parser.add_argument('--data_dir', type=str, default='./cifar')
    parser.add_argument('--result_dir', type=str, default='./results')
    
    return parser.parse_args()