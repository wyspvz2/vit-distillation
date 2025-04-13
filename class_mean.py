import torch
from collections import defaultdict

@torch.no_grad()
def compute_class_layer_mean_feature(teacher_model, dataloader, target_layer, hook_dict, target_class, device='cuda'):
    """
    针对某一类别，计算其在教师模型指定层上的类别均值特征
    Args:
        teacher_model: ViT-Large 教师模型
        dataloader: 数据加载器
        target_layer: hook中保存的目标层名称
        hook_dict: hook保存的中间层输出
        target_class: 目标类别（int）
        device: 运行设备
    Returns:
        mean_feature: [197, 1024] tensor，指定类的均值特征
    """
    teacher_model.eval()
    all_features = []

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        # 清除hook缓存
        hook_dict.clear()

        # 前向传播触发hook
        _ = teacher_model(images)

        # 提取特征：[B, 197, 1024]
        features = hook_dict[target_layer]  # 从hook中获取

        for feat, label in zip(features, labels):
            if label.item() == target_class:
                all_features.append(feat.cpu())  # feat 是 [197, 1024]

    if not all_features:
        print(f"[Warning] No samples found for class {target_class}")
        return None

    all_features = torch.stack(all_features, dim=0)  # [N, 197, 1024]
    mean_feature = all_features.mean(dim=0)  # [197, 1024]

    return mean_feature
