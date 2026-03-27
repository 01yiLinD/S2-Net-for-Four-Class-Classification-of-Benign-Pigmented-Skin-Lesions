import os
import json
import math
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
from collections import Counter
from torchvision import transforms
from torchvision.transforms import v2 
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, default_collate


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"随机种子已设置为 {seed}")


def load_box(root_dir):
    box_dict = {}
    count = 0
    
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".json"):
                json_path = os.path.join(root, file)
                file_stem = os.path.splitext(file)[0]

                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                    
                    boxes = []
                    
                    if isinstance(data, list):
                        for item in data:
                            if "bbox_xyxy" in item:
                                boxes.append(item["bbox_xyxy"])
                    
                    if len(boxes) > 0:
                        box_dict[file_stem] = boxes
                        count += 1
                except Exception as e:
                    print(f"Error reading {json_path}: {e}")
    
    print(f"成功加载了 {len(box_dict)} 个文件的 BBox 数据。")
    return box_dict


def draw_box_mask(width, height, boxes):
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)

    if (boxes is not None) and len(boxes) > 0:
        for box in boxes:
            x1, y1 = int(math.floor(box[0] * width)), int(math.floor(box[1] * height))
            x2, y2 = int(math.ceil(box[2] * width)), int(math.ceil(box[3] * height))
            draw.rectangle([x1, y1, x2, y2], fill=255)
    return mask


class SkinMoleDataset(Dataset):
    def __init__(self, data_dir, box_dict, split="train", app_transform=None, geo_transform=None):
        self.data_dir = os.path.join(data_dir, split)
        self.box_dict = box_dict if box_dict is not None else {}
        self.split = split
        self.app_transform = app_transform
        self.geo_transform = geo_transform
        self.img_paths = []
        self.labels = []
        self.class_map = {
            "compound": 0,   
            "junctional": 1,  
            "dermal": 2,      
            "seborrheic": 3  
        }

        self.idx_to_class = {v: k for k, v in self.class_map.items()}
        self.zh_names = {
            "compound": "fuhe",
            "junctional": "jiaojie",
            "dermal": "pinei",
            "seborrheic": "zhiyi"
        }
        self.filenames = []
        
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"can't find {self.data_dir}")

        files = sorted(os.listdir(self.data_dir))
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                base_name = os.path.splitext(f)[0]
                label_str = base_name.split('_')[-1]

                if label_str in self.class_map:
                    img_path = os.path.join(self.data_dir, f)
                    label = self.class_map[label_str]
                    self.img_paths.append(img_path)
                    self.labels.append(label)
                    self.filenames.append(base_name)
                else:
                    pass
        
        print(f"{self.split} load successfully, len: {len(self.img_paths)}")


    def __getitem__(self, index):
        img_path = self.img_paths[index]
        base_name = self.filenames[index]
        img = Image.open(img_path).convert("RGB")
        label = self.labels[index]

        w, h = img.size
        abs_boxes = self.box_dict.get(base_name, [])
        norm_boxes = []
        if abs_boxes:
            for box in abs_boxes:
                nx1, ny1 = box[0] / w, box[1] / h
                nx2, ny2 = box[2] / w, box[3] / h
                norm_boxes.append([nx1, ny1, nx2, ny2])
        
        mask = draw_box_mask(width=w, height=h, boxes=norm_boxes)
        img_t = transforms.functional.to_tensor(img)
        mask_t = transforms.functional.to_tensor(mask)

        combined_t = torch.cat([img_t, mask_t], dim=0)

        if self.geo_transform:
            combined_t = self.geo_transform(combined_t)
        
        img_t_trans = combined_t[:3, :, :] # RGB
        mask_t_trans = combined_t[3:, :, :] # mask

        img_transformed = transforms.functional.to_pil_image(img_t_trans)
        mask_tensor = mask_t_trans

        if self.app_transform:
            img_final = self.app_transform(img_transformed)
        else:
            img_final = transforms.ToTensor()(img_transformed)

        mask_tensor = torch.nn.functional.adaptive_max_pool2d(
            mask_tensor.unsqueeze(0), output_size=(14, 14)
        ).squeeze(0)

        mask_tensor = (mask_tensor > 0.5).float()
        if mask_tensor.sum() < 1e-6:
            mask_tensor = torch.ones(1, 14, 14)

        return img_final, label, mask_tensor


    def __len__(self):
        return len(self.labels)


def print_statistics_table(train_ds, val_ds, test_ds):
    train_counts = Counter(train_ds.labels)
    val_counts = Counter(val_ds.labels)
    test_counts = Counter(test_ds.labels)

    sorted_indices = sorted(train_ds.idx_to_class.keys())

    print("\n" + "="*75)
    header = f"{'Category Name':<15} | {'Train':<8} | {'Validation':<10} | {'Test':<8} | {'Total':<8}"
    print(header)
    print("-" * 75)

    total_train = 0
    total_val = 0
    total_test = 0

    for idx in sorted_indices:
        eng_name = train_ds.idx_to_class[idx]
        zh_name = train_ds.zh_names.get(eng_name, eng_name) 
        
        display_name = f"{zh_name}: {idx}"

        c_train = train_counts.get(idx, 0)
        c_val = val_counts.get(idx, 0)
        c_test = test_counts.get(idx, 0)
        c_row_total = c_train + c_val + c_test

        total_train += c_train
        total_val += c_val
        total_test += c_test

        print(f"{display_name:<15} | {c_train:<8} | {c_val:<10} | {c_test:<8} | {c_row_total:<8}")

    print("-" * 75) 
    grand_total = total_train + total_val + total_test
    print(f"{'total':<15} | {total_train:<8} | {total_val:<10} | {total_test:<8} | {grand_total:<8}")
    print("="*75 + "\n")


def get_data_loaders(data_dir="cropped_lesions_padded", json_dir="priv_json", use_mixup=False, mixup_prob=0.5, batch_size=64):
    box_dict = load_box(json_dir)
    train_geo_transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
    ])

    train_app_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    eval_geo_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])

    eval_app_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    data_root = data_dir 
    if not os.path.exists(data_root):
        raise FileExistsError(f"{data_root} doesn't exist")

    try:
        train_dataset = SkinMoleDataset(data_dir=data_root, box_dict=box_dict, split="train", app_transform=train_app_transform, geo_transform=train_geo_transform)
        val_dataset = SkinMoleDataset(data_dir=data_root, box_dict=None, split="val", app_transform=eval_app_transform, geo_transform=eval_geo_transform)
        test_dataset = SkinMoleDataset(data_dir=data_root, box_dict=None, split="test", app_transform=eval_app_transform, geo_transform=eval_geo_transform)
 
        print_statistics_table(train_dataset, val_dataset, test_dataset)

        print("\n--- set WeightedRandomSampler ---")
        train_targets = train_dataset.labels
        class_counts = Counter(train_targets)
        if len(class_counts) > 0:
            class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
            print(f"class_weights: {class_weights}")
            sample_weights = [class_weights[t] for t in train_targets]
            sample_weights = torch.DoubleTensor(sample_weights)
            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        else:
            sampler = None
    
    except FileNotFoundError:
        print("Warning: Dataset path not found, skipping dataset loading part for demo.")
        return None, None, None

    def custom_collate(batch):
        imgs, labels, masks = zip(*batch)
        imgs = torch.stack(imgs)
        labels = torch.tensor(labels).long().view(-1)
        masks = torch.stack(masks) # (B, 1, 14, 14)
        return imgs, labels, masks

    final_collate_fn = custom_collate

    if use_mixup:
        print(f"--- MixUp/CutMix is open (probability: {mixup_prob:.0%}) ---")
        num_classes = 4
        cutmix = v2.CutMix(num_classes=num_classes, alpha=0.2)
        mixup = v2.MixUp(num_classes=num_classes, alpha=0.2)
        cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

        def mixup_collate_fn(batch):
            imgs, labels, masks = default_collate(batch)
            if random.random() < mixup_prob:
                imgs, labels = cutmix_or_mixup(imgs, labels)
            return imgs, labels, masks
        
        final_collate_fn = mixup_collate_fn
    else:
        print("--- MixUp/CutMix is closed ---")
        final_collate_fn = custom_collate
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=8,
        collate_fn=final_collate_fn 
    )

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=custom_collate)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=custom_collate)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    try:
        train_loader, val_loader, test_loader = get_data_loaders(
            data_dir="cropped_lesions_padded",
            json_dir="priv_json",
            use_mixup=False,
            mixup_prob=0.5
        )
        
        if train_loader:
            print("\n[Test Output Shape]")
            for i, (imgs, labels, masks) in enumerate(train_loader):
                status = "Raw"
                if labels.ndim > 1:
                    status = "MixUp/CutMix Triggered!"
                
                print(f"Batch {i}: {status}")
                print(f"  Img: {imgs.shape}")
                print(f"  Label: {labels.shape}")
                print(f"  Mask: {masks.shape}")
                
                if i >= 1:
                    break
                
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"error: {e}")