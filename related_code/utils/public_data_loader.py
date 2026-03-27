import os
import math
import torch
import random
import numpy as np
import glob
import json
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw
from collections import Counter
from torchvision import transforms
from torchvision.transforms import v2 
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, default_collate
from sklearn.model_selection import train_test_split
from related_code.utils.SSI import FDA_source_to_target_np


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
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
                    print(f"error in reading {json_path}: {e}")
    
    print(len(box_dict))

    return box_dict


def draw_box_mask(width, height, boxes):
    mask = Image.new('L', (width, height), 0) # all zero
    draw = ImageDraw.Draw(mask)

    if (boxes is not None) and len(boxes) > 0:
        for box in boxes:
            x1, y1 = int(math.floor(box[0] * width)), int(math.floor(box[1] * height))
            x2, y2 = int(math.ceil(box[2] * width)), int(math.ceil(box[3] * height))
            draw.rectangle([x1, y1, x2, y2], fill=255)

    return mask


class SkinMoleDataset(Dataset):
    def __init__(self, samples, mode="train", fda_beta=0.01, box_dict=None, app_transform=None, geo_transform=None):
        self.samples = samples
        self.box_dict = box_dict if box_dict is not None else  {}
        self.app_transform = app_transform
        self.geo_transform = geo_transform
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

        # fft
        self.mode = mode
        self.fda_beta = fda_beta
        self.priv_imgs_by_class = {0: [], 1: [], 2:[], 3: []}
        if mode == "train":
            self.priv_dir = "cropped_lesions_padded/train"
            if os.path.exists(self.priv_dir):
                for f_name in os.listdir(self.priv_dir):
                    img_path = os.path.join(self.priv_dir, f_name)
                    for k, v in self.class_map.items():
                        if k in f_name:
                            self.priv_imgs_by_class[v].append(img_path)
                            break


    def __getitem__(self, index):
        img_path, label = self.samples[index]
        
        file_stem = os.path.splitext(os.path.basename(img_path))[0]
        
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # img = Image.new('RGB', (224, 224), (0, 0, 0))

        orig_w, orig_h = img.size    
        candidate_targets = self.priv_imgs_by_class.get(label, [])

        # ======================== SSI ========================
        if self.mode == "train" and len(candidate_targets) > 0:
            # print("SSI启用")
            tar_path = random.choice(candidate_targets)
            tar_img = Image.open(tar_path).convert("RGB")

            resize_h, resize_w = 256, 256
            img = img.resize((resize_w, resize_h), Image.BICUBIC)
            tar_img = tar_img.resize((resize_w, resize_h), Image.BICUBIC)

            src_np = np.asarray(img, np.float32).transpose((2, 0, 1))
            tar_np = np.asarray(tar_img, np.float32).transpose((2, 0, 1))

            src_in_trg_np = FDA_source_to_target_np(src_np, tar_np, L=self.fda_beta)
            src_in_trg_np = np.clip(src_in_trg_np, 0, 255).astype(np.uint8).transpose((1, 2, 0))
            img = Image.fromarray(src_in_trg_np)
            
            current_w, current_h = resize_w, resize_h
        elif self.mode in ["val", "test"]:
            current_w, current_h = orig_w, orig_h
        else:
            current_w, current_h = orig_w, orig_h
        # =====================================================

        # ======================== Box & Mask ========================
        abs_boxes = self.box_dict.get(file_stem, [])
        
        norm_boxes = []
        if abs_boxes:
            for box in abs_boxes:
                nx1, ny1, nx2, ny2 = box[0] / orig_w, box[1] / orig_h, box[2] / orig_w, box[3] / orig_h
                norm_boxes.append([nx1, ny1, nx2, ny2])

        mask = draw_box_mask(width=current_w, height=current_h, boxes=norm_boxes)
        img_t = transforms.functional.to_tensor(img)
        mask_t = transforms.functional.to_tensor(mask)

        combined_t = torch.cat([img_t, mask_t], dim=0)

        if self.geo_transform:
            combined_t = self.geo_transform(combined_t)

        img_t_trans = combined_t[:3, :, :] # RGB
        mask_t_trans = combined_t[3:, :, :] # Mask

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
        return len(self.samples)


    @property
    def labels(self):
        return [s[1] for s in self.samples]


def scan_and_stratified_split(root_dir, val_ratio=0.1, test_ratio=0.2, seed=42):
    class_map = {
        "compound": 0,
        "junctional": 1,
        "dermal": 2,
        "seborrheic": 3
    }
    folder_map = {
        "ISIC-images-fuhe": "compound",
        "ISIC-images-jiaojie": "junctional",
        "ISIC-images-pinei": "dermal",
        "ISIC-images-zhiyi": "seborrheic"
    }

    all_samples = [] 
    all_labels = []

    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"Root directory not found: {root_dir}")

    for folder_name, class_key in folder_map.items():
        if class_key not in class_map: continue
        label = class_map[class_key]
        folder_path = os.path.join(root_dir, folder_name)
        
        if not os.path.isdir(folder_path):
            continue

        for filename in os.listdir(folder_path):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                all_samples.append((os.path.join(folder_path, filename), label))
                all_labels.append(label)

    if not all_samples:
        raise ValueError("Data not found.")

    train_val_samples, test_samples, train_val_labels, _ = train_test_split(
        all_samples, all_labels, 
        test_size=test_ratio, 
        stratify=all_labels, 
        random_state=seed
    )

    relative_val_ratio = val_ratio / (1 - test_ratio)
    
    train_samples, val_samples = train_test_split(
        train_val_samples, 
        test_size=relative_val_ratio, 
        stratify=train_val_labels, 
        random_state=seed
    )

    return train_samples, val_samples, test_samples


def print_statistics(train_ds, val_ds, test_ds):
    print("\n" + "="*60)
    print(f"{'类别':<10} | {'Train':<6} | {'Val':<6} | {'Test':<6} | {'Total':<6}")
    print("-" * 60)
    
    t_c = Counter(train_ds.labels)
    v_c = Counter(val_ds.labels)
    te_c = Counter(test_ds.labels)
    
    total_train = total_val = total_test = 0
    
    for idx in range(4):
        name = train_ds.zh_names[train_ds.idx_to_class[idx]]
        row_tot = t_c[idx] + v_c[idx] + te_c[idx]
        print(f"{name:<10} | {t_c[idx]:<6} | {v_c[idx]:<6} | {te_c[idx]:<6} | {row_tot:<6}")
        total_train += t_c[idx]
        total_val += v_c[idx]
        total_test += te_c[idx]
        
    print("-" * 60)
    print(f"{'总计':<10} | {total_train:<6} | {total_val:<6} | {total_test:<6} | {total_train+total_val+total_test:<6}")
    print("="*60 + "\n")


def get_data_loaders(data_dir, json_dir, batch_size=64, use_mixup=False, mixup_prob=0.5):
    full_box_dict = load_box(json_dir)

    train_geo_transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180),
    ])

    train_app_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    eval_geo_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])

    eval_app_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    try:
        train_list, val_list, test_list = scan_and_stratified_split(
            data_dir, val_ratio=0.1, test_ratio=0.2
        )
    except Exception as e:
        print(f"Error: {e}")
        return None, None, None

    # 2. set Dataset and print statistics
    train_ds = SkinMoleDataset(train_list, box_dict=full_box_dict, geo_transform=train_geo_transform, app_transform=train_app_transform, mode="train")
    val_ds = SkinMoleDataset(val_list, box_dict=None, geo_transform=eval_geo_transform, app_transform=eval_app_transform, mode="val")
    test_ds = SkinMoleDataset(test_list, box_dict=None, geo_transform=eval_geo_transform, app_transform=eval_app_transform, mode="test")
    
    print_statistics(train_ds, val_ds, test_ds)

    # 3. WeightedRandomSampler (only for train)
    print("---  WeightedRandomSampler  ---")
    train_targets = train_ds.labels
    class_counts = Counter(train_targets)
    weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = [weights[t] for t in train_targets]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    # 4. MixUp Collate
    def custom_collate(batch):
        imgs, labels, masks = zip(*batch)
        imgs = torch.stack(imgs)
        labels = torch.tensor(labels).long().view(-1)
        masks = torch.stack(masks) # (B, 1, 14, 14)
        return imgs, labels, masks

    final_collate = custom_collate

    if use_mixup:
        print(f"--- MixUp is open (Prob: {mixup_prob}) ---")
        cutmix = v2.CutMix(num_classes=4, alpha=0.2)
        mixup = v2.MixUp(num_classes=4, alpha=0.2)
        choice = v2.RandomChoice([cutmix, mixup])
        
        def mixup_collate(batch):
            imgs, labels, masks = custom_collate(batch)
            if random.random() < mixup_prob:
                imgs, labels = choice(imgs, labels)
            return imgs, labels, masks
        final_collate = mixup_collate
    else:
        print("--- MixUp/CutMix is closed ---")

    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        sampler=sampler,
        # shuffle=True,
        num_workers=8,
        collate_fn=final_collate
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=custom_collate)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=custom_collate)

    return train_loader, val_loader, test_loader