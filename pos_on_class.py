import torch
import torch.nn.functional as F

@torch.no_grad()
def evaluate_cosine_similarity_with_class_mean(student_model, dataloader, target_class, 
                                                student_layer_name, class_mean_feature, 
                                                hook_dict, cap_module, device='cuda'):
    """
    评估指定类别样本在学生模型中生成的伪教师特征 与 教师模型类均值特征之间的余弦相似度
    Args:
        student_model: 学生ViT模型
        dataloader: 数据加载器
        target_class: 目标类别（int）
        student_layer_name: 学生模型中注册hook的层名称
        class_mean_feature: [197, 1024] tensor，教师该类的类均值特征
        hook_dict: forward hook缓存
        cap_module: CrossAttentionProjection 模块
        device: 运行设备
    Returns:
        avg_cosine_similarity: 该类所有样本的平均相似度
        cosine_list: 每个样本的相似度列表
    """
    student_model.eval()
    cap_module.eval()
    cosine_list = []

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        hook_dict.clear()
        _ = student_model(images)  # 前向传播触发hook

        features = hook_dict[student_layer_name]  # [B, 197, 768] 学生中间层输出

        for i in range(images.size(0)):
            if labels[i].item() != target_class:
                continue

            feat = features[i].unsqueeze(0)  # [1, 197, 768]
            pseudo_teacher_feat = cap_module(feat, class_mean_feature.to(device).unsqueeze(0))  # [1, 197, 1024]

            # 和类均值特征计算cosine similarity（沿 patch token 维度平均）
            pseudo_mean = pseudo_teacher_feat.mean(dim=1)  # [1, 1024]
            class_mean = class_mean_feature.mean(dim=0).unsqueeze(0)  # [1, 1024]

            cosine_sim = F.cosine_similarity(pseudo_mean, class_mean).item()
            cosine_list.append(cosine_sim)

    if not cosine_list:
        print(f"[Warning] No samples found for class {target_class}")
        return None, []

    avg_cos = sum(cosine_list) / len(cosine_list)
    return avg_cos, cosine_list
