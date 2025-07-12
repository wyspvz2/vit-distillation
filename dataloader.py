import torch 
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode, RandomResizedCrop
import os
import random
import traceback

class PairedTransform:
    def __init__(self, size=(224, 224), mean=None, std=None):
        self.size = size
        self.mean = mean or [0.485, 0.456, 0.406]
        self.std = std or [0.229, 0.224, 0.225]

    def _ensure_rgb_image(self, img, name=''):
        if isinstance(img, Image.Image):
            print(f"[Transform] {name} is PIL.Image.Image")
            return img.convert('RGB')
        elif isinstance(img, np.ndarray):
            print(f"[Transform] {name} is np.ndarray")
            return Image.fromarray(img).convert('RGB')
        elif hasattr(img, 'numpy'):
            print(f"[Transform] {name} is tensor-like")
            return Image.fromarray(img.numpy()).convert('RGB')
        else:
            raise TypeError(f"[Transform] Unsupported input type for {name}: {type(img)}")


    def pil_to_tensor(self, img):
        try:
            arr = np.array(img)
            print(f"[Transform] pil_to_tensor input array dtype: {arr.dtype}, shape: {arr.shape}")
            if arr.dtype != np.uint8:
                arr = arr.astype(np.uint8)
                print(f"[Transform] Converted array to uint8")
            # 修复 torch.from_numpy 报错
            tensor = torch.tensor(arr.copy(), dtype=torch.float32).permute(2, 0, 1).div(255)
            print(f"[Transform] Converted to tensor shape: {tensor.shape}, dtype: {tensor.dtype}")
            return tensor
        except Exception as e:
            print("[Transform] Exception in pil_to_tensor:")
            traceback.print_exc()
            raise e

    def __call__(self, opt, sar):
        print("[Transform] Start __call__")
        opt = self._ensure_rgb_image(opt, 'opt')
        sar = self._ensure_rgb_image(sar, 'sar') if sar is not None else None
        print(f"[Transform] After ensure_rgb_image: opt size={opt.size}; sar size={sar.size if sar else None}")

        i, j, h, w = RandomResizedCrop.get_params(opt, scale=(0.2, 1.0), ratio=(3/4, 4/3))
        print(f"[Transform] Crop params: i={i}, j={j}, h={h}, w={w}")
        opt = F.resized_crop(opt, i, j, h, w, self.size, InterpolationMode.BICUBIC)
        sar = F.resized_crop(sar, i, j, h, w, self.size, InterpolationMode.BICUBIC) if sar is not None else None
        print(f"[Transform] After resized_crop: opt size={opt.size}; sar size={sar.size if sar else None}")

        if random.random() < 0.5:
            print("[Transform] Applying horizontal flip")
            opt = F.hflip(opt)
            if sar is not None:
                sar = F.hflip(sar)

        print("[Transform] Converting opt to tensor")
        opt = self.pil_to_tensor(opt)
        opt = F.normalize(opt, mean=self.mean, std=self.std)

        if sar is not None:
            print("[Transform] Converting sar to tensor")
            sar = self.pil_to_tensor(sar)
            sar = F.normalize(sar, mean=self.mean, std=self.std)
            print("[Transform] Finished transform, returning opt and sar")
            return opt, sar

        print("[Transform] Finished transform, returning opt only")
        return opt




     


class PairedModalDataset(Dataset):
    def __init__(self, root_mod1, root_mod2, transform=None):
        self.root_mod1 = root_mod1
        self.root_mod2 = root_mod2
        self.transform = transform

        self.classes = sorted(os.listdir(root_mod1))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        self.samples = []
        for cls in self.classes:
            mod1_cls_dir = os.path.join(root_mod1, cls)
            mod2_cls_dir = os.path.join(root_mod2, cls)

            files_mod1 = sorted(os.listdir(mod1_cls_dir))
            files_mod2 = sorted(os.listdir(mod2_cls_dir))

            mod1_dict = {f[3:]: f for f in files_mod1}
            mod2_dict = {f[3:]: f for f in files_mod2}

            common_keys = set(mod1_dict.keys()) & set(mod2_dict.keys())

            for key in common_keys:
                mod1_path = os.path.join(mod1_cls_dir, mod1_dict[key])
                mod2_path = os.path.join(mod2_cls_dir, mod2_dict[key])
                if os.path.isfile(mod1_path) and os.path.isfile(mod2_path):
                    self.samples.append((mod1_path, mod2_path, self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        opt_path, sar_path, target = self.samples[idx]
        
        # 读取图片并转为RGB
        opt = Image.open(opt_path).convert('RGB')
        sar = Image.open(sar_path).convert('RGB')
        print(f"[Dataset] Loaded images at idx={idx}: opt type={type(opt)}, mode={opt.mode}; sar type={type(sar)}, mode={sar.mode}")

        # 应用transform，返回tensor
        if self.transform:
            opt, sar = self.transform(opt, sar)
        else:
            # 如果没有transform，手动转换为tensor
            opt = F.to_tensor(opt)
            sar = F.to_tensor(sar)

        print(f"[Dataset] After transform idx={idx}: opt shape={getattr(opt, 'shape', None)}; sar shape={getattr(sar, 'shape', None)}")

        # 确保target是tensor（整型）
        target = torch.tensor(target, dtype=torch.long)

        return opt, sar, target



def get_paired_dataloader_train(root_mod1='/home/wys/CMKD/dataset/opt123/train',
                                root_mod2='/home/wys/CMKD/dataset/sar123/train',
                                batch_size=64, num_workers=0):
    transform = PairedTransform(size=(224, 224))
    dataset_train = PairedModalDataset(root_mod1, root_mod2, transform=transform)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return loader_train, dataset_train


def get_paired_dataloader_val(root_mod1='/home/wys/CMKD/dataset/opt123/val',
                              root_mod2='/home/wys/CMKD/dataset/sar123/val',
                              batch_size=64, num_workers=0):
    transform = PairedTransform(size=(224, 224))
    dataset_eval = PairedModalDataset(root_mod1, root_mod2, transform=transform)
    loader_eval = DataLoader(dataset_eval, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return loader_eval, dataset_eval
