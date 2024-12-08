# util/metrics.py
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def calculate_metrics(model, data_loader, criterion=None, device='cpu'):
    """
    计算模型在验证集或测试集上的评估指标。

    参数：
    - model: 评估的模型
    - data_loader: 验证集或测试集的 DataLoader
    - criterion: 损失函数（可选），用于计算平均损失
    - device: 运行设备（'cpu' 或 'cuda'）

    返回：
    - metrics: 包含准确率、精确率、召回率、F1 分数、AUC 和（可选）平均损失的字典
    """
    model.to(device)
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []
    total_loss = 0

    with torch.no_grad():
        for batch in data_loader:
            inputs = batch['video'].to(device)
            labels = batch['label'].to(device)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # 获取属于正类的概率

            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            # 如果提供了损失函数，计算并累计损失
            if criterion is not None:
                loss = criterion(outputs, labels)
                total_loss += loss.item()

    # 计算各项评估指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else float('nan')  # 防止单一类别 AUC 计算异常

    # 整理评估结果
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc
    }

    # 如果有 criterion，计算并加入平均损失
    if criterion is not None:
        metrics['loss'] = total_loss / len(data_loader)

    return metrics
