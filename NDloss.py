import torch
import torch.nn.functional as F

def compute_projection_l2_loss(student_features, teacher_features, labels, num_classes, device='cuda'):
    """
    计算教师和伪教师投影后的L2损失，并对每个类别求平均，再对所有类别的损失求均值
    Args:
        student_features: 伪教师特征，形状为 [B, 197, 1024]
        teacher_features: 教师特征，形状为 [B, 197, 1024]
        labels: 每个样本的标签，形状为 [B]
        num_classes: 类别数 C
        device: 运行设备
    Returns:
        avg_loss: 平均的L2损失
    """
    student_features = student_features.to(device)  # [B, 197, 1024]
    teacher_features = teacher_features.to(device)  # [B, 197, 1024]
    labels = labels.to(device)  # [B]

    total_loss = 0.0
    for k in range(num_classes):
        # 获取当前类别 k 的样本索引
        class_indices = (labels == k).nonzero(as_tuple=True)[0]  # 所有属于类别 k 的样本索引
        
        if len(class_indices) == 0:
            continue
        
        # 对于属于当前类别 k 的所有样本
        class_student_features = student_features[class_indices]  # 形状 [N_k, 197, 1024]
        class_teacher_features = teacher_features[class_indices]  # 形状 [N_k, 197, 1024]

        # 计算L2损失
        l2_loss = torch.norm(class_student_features - class_teacher_features, p=2, dim=-1)  # [N_k, 197]
        
        # 计算教师和伪教师特征的L2范数
        student_norm = torch.norm(class_student_features, p=2, dim=-1)  # [N_k, 197]
        teacher_norm = torch.norm(class_teacher_features, p=2, dim=-1)  # [N_k, 197]

        # 归一化损失
        max_norm = torch.max(student_norm, teacher_norm)  # [N_k, 197]
        normalized_l2_loss = l2_loss / max_norm  # [N_k, 197]

        # 计算当前类别 k 的损失
        class_loss = normalized_l2_loss.mean()  # 平均化该类别的L2损失
        total_loss += class_loss  # 将该类别的损失累加到总损失

    # 计算所有类别的平均损失
    avg_loss = total_loss / num_classes

    return avg_loss
