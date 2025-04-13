import torch
import torch.nn.functional as F

@torch.no_grad()
def project_teacher_mean_to_student(pseudo_teacher_features, teacher_mean_feature, device='cuda'):
    """
    计算伪教师特征到教师模型指定类别均值特征的投影
    Args:
        pseudo_teacher_features: 伪教师特征，形状为 [B, 197, 1024]，其中B为批次大小
        teacher_mean_feature: 教师模型的类别均值特征，形状为 [197, 1024]
        device: 运行设备
    Returns:
        projected_features_S: 投影后的特征，形状为 [B, 197, 1024]
    """
    pseudo_teacher_features = pseudo_teacher_features.to(device)  # [B, 197, 1024]
    teacher_mean_feature = teacher_mean_feature.to(device)  # [197, 1024]

    # 扩展教师的类别均值特征到批次大小 B
    teacher_mean_expanded = teacher_mean_feature.unsqueeze(0).expand(pseudo_teacher_features.size(0), -1, -1)  # [B, 197, 1024]

    # 通过规范化特征，进行投影
    pseudo_teacher_features_normalized = F.normalize(pseudo_teacher_features, p=2, dim=-1)  # [B, 197, 1024]
    teacher_mean_expanded_normalized = F.normalize(teacher_mean_expanded, p=2, dim=-1)  # [B, 197, 1024]

    # 计算学生特征到教师均值特征的投影，投影使用点乘
    projected_features_S = torch.bmm(pseudo_teacher_features_normalized, teacher_mean_expanded_normalized.transpose(1, 2))  # [B, 197, 197]

    return projected_features_S

@torch.no_grad()
def project_teacher_features_to_class_mean(teacher_features, teacher_mean_feature, device='cuda'):
    """
    计算教师特征到教师模型指定类别均值特征的投影
    Args:
        teacher_features: 教师特征，形状为 [B, 197, 1024]，其中B为批次大小
        teacher_mean_feature: 教师模型的类别均值特征，形状为 [197, 1024]
        device: 运行设备
    Returns:
        projected_features_T: 投影后的特征，形状为 [B, 197, 1024]
    """
    teacher_features = teacher_features.to(device)  # [B, 197, 1024]
    teacher_mean_feature = teacher_mean_feature.to(device)  # [197, 1024]

    # 扩展教师的类别均值特征到批次大小 B
    teacher_mean_expanded = teacher_mean_feature.unsqueeze(0).expand(teacher_features.size(0), -1, -1)  # [B, 197, 1024]

    # 通过规范化特征，进行投影
    teacher_features_normalized = F.normalize(teacher_features, p=2, dim=-1)  # [B, 197, 1024]
    teacher_mean_expanded_normalized = F.normalize(teacher_mean_expanded, p=2, dim=-1)  # [B, 197, 1024]

    # 计算教师特征到教师均值特征的投影，投影使用点乘
    projected_features_T = torch.bmm(teacher_features_normalized, teacher_mean_expanded_normalized.transpose(1, 2))  # [B, 197, 197]

    return projected_features_T
