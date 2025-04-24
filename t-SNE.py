import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# 假设你已经训练好的模型（ViT）和验证集
# 输出的是每个样本的特征（通常是模型的中间层输出）
def plot_tsne(model, dataloader, device):
    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.numpy()
            
            # 获取模型的中间层输出特征
            outputs = model(inputs)
            
            # 假设输出的特征是最后一层的特征（例如：features = model.last_layer_output）
            # 这里我们假设你已经提取了模型的特征
            features = outputs.cpu().numpy()
            
            all_features.append(features)
            all_labels.append(labels)

    # 合并所有特征和标签
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # 使用t-SNE降维
    tsne = TSNE(n_components=2, random_state=42)
    reduced_features = tsne.fit_transform(all_features)
    
    # 使用LabelEncoder处理标签
    le = LabelEncoder()
    all_labels = le.fit_transform(all_labels)
    
    # 绘制t-SNE图
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=all_labels, cmap='tab10', s=30)
    plt.colorbar(scatter)
    plt.title("t-SNE visualization")
    plt.show()

# 使用时直接传入模型、dataloader和设备
# plot_tsne(model, val_loader, device)
