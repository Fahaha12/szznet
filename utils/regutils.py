import matplotlib.pyplot as plt
import torch
import os
from sklearn.metrics import r2_score, mean_absolute_error
import re

def calculate_metrics(y_true, y_pred):
    """计算 R² 分数和均方误差 (MAE)"""
    # 确保输入是 PyTorch 张量
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()  # 转换为 NumPy 数组
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()  # 转换为 NumPy 数组
    
    r2 = r2_score(y_true, y_pred)  # 计算 R² 分数
    mae = mean_absolute_error(y_true, y_pred)  # 计算 MAE
    return r2, mae
    
def plot_loss(train_losses, val_losses, r2_scores, mae_scores, out_path, epoch):
    """ 绘制训练和验证损失图以及 R² 分数和 MAE """
    plt.figure(figsize=(12, 8))

    # 绘制训练和验证损失
    plt.subplot(2, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # 绘制 R² 分数
    plt.subplot(2, 2, 2)
    plt.plot(range(1, len(r2_scores) + 1), r2_scores, label='R² Score', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('R² Score')
    plt.title('Validation R² Score')
    plt.legend()
    plt.grid(True)

    # 绘制 MAE
    plt.subplot(2, 2, 3)
    plt.plot(range(1, len(mae_scores) + 1), mae_scores, label='MAE', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('Validation MAE')
    plt.legend()
    plt.grid(True)

    # 保存图像
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, f'loss_plot_epoch_{epoch}.jpg'))
    plt.close()


def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return epoch    

def find_latest_checkpoint(out_path):
    # 获取所有符合模式的检查点文件
    checkpoint_files = [f for f in os.listdir(out_path) if re.match(r'latest_model_checkpoint_epoch_\d+\.pth', f)]
    if not checkpoint_files:
        return None

    # 找到最新的检查点文件
    latest_epoch = -1
    latest_checkpoint = None
    for file in checkpoint_files:
        # 提取epoch编号
        match = re.search(r'latest_model_checkpoint_epoch_(\d+)\.pth', file)
        if match:
            epoch = int(match.group(1))
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest_checkpoint = file

    return os.path.join(out_path, latest_checkpoint) if latest_checkpoint else None    