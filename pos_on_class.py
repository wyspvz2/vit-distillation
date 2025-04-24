def project_teacher_mean_to_student(pseudo_teacher_features, teacher_mean_feature, labels):
    """
    将伪教师特征投影到对应类别的教师均值特征上，输出保持 [B, 197, 1024]
    """
    selected_teacher_means = teacher_mean_feature[labels]  # [B, 197, 1024]
    # 单位化教师均值特征向量（逐 token 单位化）
    mean_unit = F.normalize(selected_teacher_means, p=2, dim=-1)  # [B, 197, 1024]

    # 计算内积 (x ⋅ u)，需要逐 token 点乘
    dot_product = torch.sum(pseudo_teacher_features * mean_unit, dim=-1, keepdim=True)  # [B, 197, 1]

    # 投影结果 = (x ⋅ u) * u
    projected_features_S = dot_product * mean_unit  # [B, 197, 1024]
    return projected_features_S


def project_teacher_features_to_class_mean(teacher_features, teacher_mean_feature, labels):
    """
    将教师特征投影到对应类别的教师均值特征上，输出保持 [B, 197, 1024]
    """
    selected_teacher_means = teacher_mean_feature[labels]  # [B, 197, 1024]
    mean_unit = F.normalize(selected_teacher_means, p=2, dim=-1)  # [B, 197, 1024]

    dot_product = torch.sum(teacher_features * mean_unit, dim=-1, keepdim=True)  # [B, 197, 1]

    projected_features_T = dot_product * mean_unit  # [B, 197, 1024]
    return projected_features_T
