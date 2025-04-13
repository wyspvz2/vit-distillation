import torch
import torch.nn as nn
import torch.optim as optim
import timm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.nn import LayerNorm
from mmcv.cnn.bricks.transformer import MultiheadAttention
import numpy as np
from torch.nn import MultiheadAttention
from torchvision import datasets
import os

def register_hooks(model, target_layers, feature_list):
    hooks = []
    for layer in target_layers:
        hook = layer.register_forward_hook(lambda module, input, output: feature_list.append(output))
        hooks.append(hook)
    return hooks


def train(student_model, teacher_model, cap_module, dct_distillation_loss, train_loader, optimizer, device,
          num_epochs=10, save_path="student_model_final.pth", distill_weight=1.0):
    teacher_model.eval()
    criterion_cls = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        student_model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            student_features, teacher_features = [], []

            # 注册 hook（student 第0~11层，teacher 第12~23层）
            student_target_layers = [student_model.blocks[n].norm1 for n in range(12)]
            teacher_target_layers = [teacher_model.blocks[n + 12].norm1 for n in range(12)]

            student_hooks = register_hooks(student_model, student_target_layers, student_features)
            teacher_hooks = register_hooks(teacher_model, teacher_target_layers, teacher_features)

            # 触发前向传播
            with torch.no_grad():
                _ = teacher_model(images)
            student_outputs = student_model(images)

            for h in student_hooks + teacher_hooks:
                h.remove()

            # 多层蒸馏损失累计
            loss_distill = 0.0
            for n in range(12):
                student_feat = student_features[n]
                teacher_feat = teacher_features[n]
                pseudo_teacher_feat = cap_module(student_feat, teacher_feat)
                loss_layer = dct_distillation_loss(pseudo_teacher_feat, teacher_feat)
                loss_distill += loss_layer

            # 分类损失
            loss_cls = criterion_cls(student_outputs, labels)

            # 总损失
            loss = loss_cls + distill_weight * loss_distill

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 准确率计算
            _, predicted = torch.max(student_outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}] | Batch {batch_idx} | "
                      f"Cls Loss: {loss_cls.item():.4f} | Distill Loss: {loss_distill.item():.4f} | Total Loss: {loss.item():.4f}")

        avg_loss = running_loss / len(train_loader)
        accuracy = correct / total
        print(f"\n==> Epoch [{epoch + 1}/{num_epochs}] DONE | Avg Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")

    # ✅ 保存模型
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(student_model.state_dict(), save_path)
    print(f"\n✅ 模型权重已保存到: {save_path}")

# Main function to initialize and train the model
def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # Force using CUDA0
    num_classes = 10

    checkpoint_path1 = '/home/ymhj/wys/weights/20250406-133657-vit_large_patch16_224-224/model_best.pth.tar'

    # Load the teacher model (ViT-Large) on CUDA0
    teacher_model = timm.create_model('vit_large_patch16_224', pretrained=False, checkpoint_path=checkpoint_path1,
                                      num_classes=num_classes).to(device)

    checkpoint_path2 = '/home/ymhj/wys/weights/20250405-115543-vit_base_patch16_224-224/model_best.pth.tar'

    # Load the student model (ViT-Base) on CUDA0
    student_model = timm.create_model('vit_base_patch16_224', pretrained=False, checkpoint_path=checkpoint_path2,
                                      num_classes=num_classes).to(device)

    # Initialize CAP module
    cap_module = CAPModule(student_dims=768, teacher_dims=1024).to(device)

    # Initialize DCT module
    dct_module = DCTDistillationModule(resolution=1024, device=device).to(device)

    # Initialize DCTDistillationLoss module with specified weights
    distillation_loss_module = DCTDistillationLoss(dct_module, distillation_weight=1.0, dct_weight=1.0)

    # Define the optimizer for the student model and CAP module
    optimizer = optim.Adam(list(student_model.parameters()) + list(cap_module.parameters()), lr=1e-4)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 强制调整图像大小为224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Assuming your dataset is in '/home/ymhj/wys/Dataset/AugmentedDataset/train' with subfolders representing classes
    train_dataset = datasets.ImageFolder(root='/home/ymhj/wys/Dataset/AugmentedDataset/train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=16)

    # Train the student model
    train(student_model, teacher_model, cap_module, distillation_loss_module, train_loader, optimizer, device, num_epochs=10,  save_path="/home/ymhj/wys/weights/check/student_model_latest.pth.tar")


if __name__ == '__main__':
    main()
