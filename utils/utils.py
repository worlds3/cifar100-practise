import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import json

def save_config(args, save_path):
    with open(save_path, 'w') as f:
        json.dump(vars(args), f, indent=4)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def create_experiment_dir(config):
    result_path = os.path.join(config.result_dir, config.exp_name)
    os.makedirs(result_path, exist_ok=True)
    return result_path

def save_training_plots(history, result_path):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.savefig(os.path.join(result_path, 'training_curves.png'))
    plt.close()