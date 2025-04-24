def register_hooks(model, target_layers, feature_list):
    hooks = []
    for layer in target_layers:
        hook = layer.register_forward_hook(lambda module, input, output: feature_list.append(output))
        hooks.append(hook)
    return hooks


def train(student_model, teacher_model, project_teacher_mean_to_student, project_teacher_features_to_class_mean,
          compute_class_means, cap_module, distillation_loss_module, compute_projection_l2_loss, train_loader,
          val_loader,
          optimizer, device,
          num_epochs=100, save_path="",
          distill_weight=1.0, loss_kd2_weight=0.1, num_classes=10,
          csv_path=""):
    teacher_model.eval()
    criterion_cls = torch.nn.CrossEntropyLoss()
    # 初始化 CSV 日志文件（包含表头）
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_acc1'])  # timm 风格表头

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
            teacher_target_layers = [teacher_model.blocks[n].norm1 for n in range(12)]

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
            loss_kd2 = 0.0
            # 在 for 循环外预先计算好指数型归一化权重
            tau = 3.0  # 温度系数，越小权重增长越快
            exp_weights = [math.exp(n / tau) for n in range(12)]
            weight_sum = sum(exp_weights)
            layer_weights = [w / weight_sum for w in exp_weights]
            for n in range(12):
                student_feat = student_features[n]
                teacher_feat = teacher_features[n]
                pseudo_teacher_feat = cap_module(student_feat, teacher_feat)
                # 层权重：e增长
                layer_weight = layer_weights[n]
                # 蒸馏损失（DCT部分）
                loss_layer = distillation_loss_module(pseudo_teacher_feat, teacher_feat)
                loss_distill += layer_weight * loss_layer  # 加权累计

                # 类别均值投影蒸馏损失
                teacher_mean_feature1 = compute_class_means(teacher_feat, labels, num_classes=10)
                projected_features_S1 = project_teacher_mean_to_student(pseudo_teacher_feat, teacher_mean_feature1,
                                                                        labels)
                projected_features_T1 = project_teacher_features_to_class_mean(teacher_feat, teacher_mean_feature1,
                                                                               labels)
                loss_proj = compute_projection_l2_loss(
                    pseudo_teacher_feat, teacher_feat,
                    projected_features_S1, projected_features_T1,
                    labels, num_classes=10
                )
                loss_kd2 += layer_weight * loss_proj  # 同样加权

            # 分类损失
            loss_cls = criterion_cls(student_outputs, labels)

            # 总损失
            loss = loss_cls + distill_weight * loss_distill + loss_kd2_weight * loss_kd2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}] | Batch {batch_idx} | "
                      f"Cls Loss: {loss_cls.item():.4f} | Distill Loss: {loss_distill.item():.4f} | kd++loss: {loss_kd2.item():.4f} | Total Loss: {loss.item():.4f}")

        # epoch 结束后：
        avg_loss = running_loss / len(train_loader)
        val_accuracy = evaluate(student_model, val_loader, device)
        print(f"\n==> Epoch [{epoch + 1}/{num_epochs}] DONE | Avg Loss: {avg_loss:.4f} | Acc@1: {val_accuracy:.2f}")

        # 写入 CSV 日志
        with open(csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_loss, val_accuracy])

    # ✅ 保存模型
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(student_model.state_dict(), save_path)
    print(f"\n✅ 模型权重已保存到: {save_path}")


# Main function to initialize and train the model
def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # Force using CUDA0
    num_classes = 10

    checkpoint_path1 = '/home/ymhj/wys/weights/20250405-224639-vit_base_patch16_224-224/model_best.pth.tar'

    # Load the teacher model (ViT-Large) on CUDA0
    teacher_model = timm.create_model('vit_base_patch16_224', pretrained=False, checkpoint_path=checkpoint_path1,
                                      num_classes=num_classes).to(device)

    checkpoint_path2 = '/home/ymhj/wys/weights/20250420-222421-vit_tiny_patch16_224_augreg_in21k-224/model_best.pth.tar'

    # Load the student model (ViT-Base) on CUDA0
    student_model = timm.create_model('vit_tiny_patch16_224.augreg_in21k', pretrained=False,
                                      checkpoint_path=checkpoint_path2,
                                      num_classes=num_classes).to(device)

    # Initialize CAP module
    cap_module = CAPModule(student_dims=192, teacher_dims=768).to(device)

    # Initialize DCT module
    dct_module = DCTDistillationModule(resolution=768, device=device).to(device)

    # Initialize DCTDistillationLoss module with specified weights
    distillation_loss_module = DCTDistillationLoss(dct_module, distillation_weight=1.0, dct_weight=1.0)

    # Define the optimizer for the student model and CAP module
    optimizer = optim.Adam(list(student_model.parameters()) + list(cap_module.parameters()), lr=1e-4)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 强制调整图像大小为224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Assuming your dataset is in '' with subfolders representing classes
    train_dataset = datasets.ImageFolder(root='', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=16)
    val_dataset = datasets.ImageFolder(root='/home/ymhj/wys/Dataset/AugmentedDataset/val', transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=16)

    # Train the student model
    # Train the student model
    train(student_model, teacher_model, project_teacher_mean_to_student, project_teacher_features_to_class_mean,
          compute_class_means, cap_module, distillation_loss_module, compute_projection_l2_loss, train_loader,
          val_loader,
          optimizer, device,
          num_epochs=100, save_path="",
          distill_weight=1.0, loss_kd2_weight=0.1, num_classes=10,
          csv_path="")

if __name__ == '__main__':
    main()
