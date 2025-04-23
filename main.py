import os
from config import get_config
from dataload.dataload import get_dataloaders
import model.resnet as module_arch
from trainer.train import train_model
from utils.utils import setup_seed, create_experiment_dir, save_training_plots, save_config
import torch
import torch.optim as optim
from torch.optim import lr_scheduler

def main():
    # 获取配置
    config = get_config()
    
    # 设置随机种子
    setup_seed(config.seed)
    
    # 创建实验目录
    result_path = create_experiment_dir(config)
    
    # 准备数据
    dataloaders, class_names = get_dataloaders(config.data_dir, config.batch_size)
    dataset_sizes = {
        'train': len(dataloaders['train'].dataset),
        'val': len(dataloaders['val'].dataset)
    }
    
    # 初始化模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = getattr(module_arch, config.model)(num_classes=len(class_names)).to(device)
    
    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer, 
        step_size=config.step_size,
        gamma=config.gamma
    )
    
    # 训练模型
    model, history = train_model(
        model=model,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=exp_lr_scheduler,
        num_epochs=config.epochs,
        result_path=result_path
    )
    # 保存训练曲线
    save_training_plots(history, result_path)
    # 保存训练超参数
    save_config(config, os.path.join(result_path, 'config.json'))

if __name__ == '__main__':
    main()