import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from utils import GradCAM, show_cam_on_image, center_crop_img
import timm
import matplotlib
matplotlib.use('Agg')

class ReshapeTransform:
    def __init__(self, model):
        input_size = model.patch_embed.img_size
        patch_size = model.patch_embed.patch_size
        self.h = input_size[0] // patch_size[0]
        self.w = input_size[1] // patch_size[1]

    def __call__(self, x):
        # remove cls token and reshape
        result = x[:, 1:, :].reshape(x.size(0), self.h, self.w, x.size(2))
        result = result.permute(0, 3, 1, 2)
        return result


def save_gradcam_image(visualization, output_path):
    """
    保存 GradCAM 热图到指定路径
    """
    plt.imshow(visualization)
    plt.axis('off')  # 关闭坐标轴
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)  # 保存图像
    plt.close()


def main():
    # 1. 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. 定义并加载模型
    checkpoint_path = ''
    model = timm.create_model('vit_base_patch16_224', pretrained=False,
                              checkpoint_path=checkpoint_path, num_classes=10)
    model.to(device)
    model.eval()

    # 3. 图像预处理
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    img_path = ""
    assert os.path.exists(img_path), f"file: '{img_path}' does not exist."
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    img = center_crop_img(img, 224)
    img_tensor = data_transform(img)
    input_tensor = torch.unsqueeze(img_tensor, dim=0).to(device)

    # 4. 循环生成每一层的 GradCAM
    output_dir = "/home/ymhj/wys/Dataset/hotmap/10"
    os.makedirs(output_dir, exist_ok=True)

    for q in range(1, 13):  # q 从 1 到 12
        print(f"Processing layer {q}...")
        target_layers = [model.blocks[-q].norm1]

        cam = GradCAM(model=model,
                      target_layers=target_layers,
                      use_cuda=True,
                      reshape_transform=ReshapeTransform(model))

        target_category = 3  # 指定类别
        grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)[0, :]
        visualization = show_cam_on_image(img / 255., grayscale_cam, use_rgb=True)

        output_path = os.path.join(output_dir, f"gradcam_output_layer_{q}.png")
        save_gradcam_image(visualization, output_path)
        print(f"Saved: {output_path}")



if __name__ == '__main__':
    main()
