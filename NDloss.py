def compute_projection_l2_loss(pseudo_teacher_features, teacher_features,
                               projected_features_S, projected_features_T, labels, num_classes):
    """
    计算教师和伪教师投影后的L2损失，并对每个类别求平均，再对所有类别的损失进行样本量加权平均
    Args:
        pseudo_teacher_features: [B, 197, d]
        teacher_features: [B, 197, d]
        projected_features_S: [B, 197, d]
        projected_features_T: [B, 197, d]
        labels: [B]
    Returns:
        avg_loss: scalar
    """
    total_loss = 0.0
    total_samples = 0

    for k in range(num_classes):
        # 获取当前类别样本索引
        class_indices = (labels == k).nonzero(as_tuple=True)[0]

        if class_indices.numel() == 0:
            continue

        # 提取该类别下的特征
        proj_S = projected_features_S[class_indices]  # [N_k, 197, d]
        proj_T = projected_features_T[class_indices]  # [N_k, 197, d]
        feat_S = pseudo_teacher_features[class_indices]  # [N_k, 197, d]
        feat_T = teacher_features[class_indices]  # [N_k, 197, d]

        # 计算每个 patch 的通道向量 L2 范数
        norm_S = torch.norm(feat_S, dim=-1)  # [N_k, 197]
        norm_T = torch.norm(feat_T, dim=-1)  # [N_k, 197]
        norm_Ts = torch.norm(proj_S, dim=-1)  # [N_k, 197]
        # 防止除零
        max_norm = torch.max(norm_S, norm_T) + 1e-6  # [N_k, 197]

        # 计算投影向量之间的 L2 距离
        diff = proj_S - proj_T  # [N_k, 197, d]
        dist = torch.norm(diff, dim=-1)  # [N_k, 197]

        # 归一化距离
        normed_dist = 1 - (norm_Ts / (max_norm + 1e-6))  # [N_k, 197]
        # 类内平均
        class_loss = normed_dist.mean()

        # 样本加权
        total_loss += class_loss * class_indices.numel()
        total_samples += class_indices.numel()

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    return avg_loss


@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  # top-1 prediction
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy * 100  # 返回百分比形式，和 timm 一致
