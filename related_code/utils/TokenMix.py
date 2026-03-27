# import math
# import os
# import random
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import cv2
# import sys

# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(os.path.dirname(current_dir))
# sys.path.append(parent_dir)

# from related_code.models import create_model
# # from data_loader import get_data_loaders
# # from transformers import ViTForImageClassification, ViTConfig


# def one_hot(x, num_classes, on_value=1., off_value=0., device='cuda'):
#     x = x.long().view(-1, 1)
#     return torch.full((x.size()[0], num_classes), off_value, device=device).scatter_(1, x, on_value)

# def mixup_target(target, num_classes, lam=1., smoothing=0.0, device='cuda'):
#     off_value = smoothing / num_classes
#     on_value = 1. - smoothing + off_value
#     y1 = one_hot(target, num_classes, on_value=on_value, off_value=off_value, device=device)
#     y2 = one_hot(target.flip(0), num_classes, on_value=on_value, off_value=off_value, device=device)
#     return y1 * lam + y2 * (1. - lam)

# # ==========================================
# # 2. 改进的 Mask 生成 (支持 Batch 独立生成)
# # ==========================================

# def generate_single_block_mask(lam, device, mask_token_num_start=1):
#     """生成单张图的 Block Mask (14x14)"""
#     width, height = 14, 14
#     mask = np.zeros(shape=(height, width), dtype=np.float32)
#     mask_ratio = 1 - lam
#     num_masking_patches = min(width * height, int(mask_ratio * width * height) + mask_token_num_start)
    
#     mask_count = 0
#     while mask_count < num_masking_patches:
#         target_area = random.uniform(2, 10) # 每次涂抹的块大小
#         aspect_ratio = random.uniform(0.3, 3.3)
#         h = int(round(math.sqrt(target_area * aspect_ratio)))
#         w = int(round(math.sqrt(target_area / aspect_ratio)))
#         if w < width and h < height:
#             top = random.randint(0, height - h)
#             left = random.randint(0, width - w)
#             for i in range(top, top + h):
#                 for j in range(left, left + w):
#                     if mask[i, j] == 0 and mask_count < num_masking_patches:
#                         mask[i, j] = 1
#                         mask_count += 1
#         if target_area > num_masking_patches - mask_count: break # 防止死循环

#     mask = torch.from_numpy(mask).to(device).unsqueeze(0).unsqueeze(0) # (1, 1, 14, 14)
#     return mask, 1 - (mask_count / (width * height))

# def generate_mask_batch(batch_size, lam, device, mask_type='block', mask_token_num_start=14):
#     """为整个 Batch 生成独立的 Mask"""
#     masks = []
#     lams = []
#     for _ in range(batch_size):
#         if mask_type == 'block':
#             m, l = generate_single_block_mask(lam, device, mask_token_num_start)
#         else: # random
#             m_flat = torch.zeros(196, device=device)
#             idx = torch.randperm(196, device=device)[:int(196 * (1-lam))]
#             m_flat[idx] = 1
#             m = m_flat.view(1, 1, 14, 14)
#             l = 1 - (len(idx)/196)
#         masks.append(m)
#         lams.append(l)
#     return torch.cat(masks, dim=0), np.mean(lams)

# # ==========================================
# # 3. 改进的 Dense Label 生成 (多层注意力融合)
# # ==========================================

# @torch.no_grad()
# def generate_dense_labels(teacher_model, images, labels, num_classes=4, top_k_layers=3):
#     """
#     利用 Teacher 模型融合最后 K 层注意力生成更稳健的 Dense Labels
#     """
#     teacher_model.eval()
#     # images: (B, 3, 224, 224)
#     outputs = teacher_model.vit(pixel_values=images, output_attentions=True)
    
#     # 1. 融合最后 K 层的 Attention Maps
#     # attentions 是一元组，每层形状 (B, Heads, 197, 197), 197 = 1 (CLS) + 196 (Patch)
#     relevant_attentions = outputs.attentions[-top_k_layers:] 
#     stack_attn = torch.stack(relevant_attentions) # (K, B, Heads, 197, 197)
    
#     # 2. 对层维度和头维度取平均
#     combined_attn = stack_attn.mean(dim=0).mean(dim=1) # (B, 197, 197)
    
#     # 3. 提取 CLS Token 对所有 Patch 的注意力 (索引 0 为 CLS)
#     cls_attn = combined_attn[:, 0, 1:] # (B, 196)
    
#     # 4. 归一化 Saliency Map (Min-Max)
#     B = cls_attn.shape[0]
#     attn_min = cls_attn.min(dim=1, keepdim=True)[0]
#     attn_max = cls_attn.max(dim=1, keepdim=True)[0]
#     cls_attn = (cls_attn - attn_min) / (attn_max - attn_min + 1e-6)
    
#     # 5. 生成空间热力图 (14x14)
#     saliency_map = cls_attn.view(B, 1, 14, 14)
    
#     # 6. 广播标签: (B, C, 1, 1) * (B, 1, 14, 14) -> (B, C, 14, 14)
#     labels_onehot = F.one_hot(labels, num_classes=num_classes).float().view(B, num_classes, 1, 1)
#     dense_labels = labels_onehot * saliency_map
    
#     return dense_labels


# def generate_mask_from_gt(gt_mask, device, mode='keep_foreground'):
#     """
#     根据 Ground Truth Mask 生成 14x14 的 Block Mask。
    
#     Args:
#         gt_mask: [B, 1, 224, 224], 值为 0 或 1
#         mode: 
#             'keep_foreground': 保留前景(A)，替换背景(B) -> Mask在背景为1
#             'swap_foreground': 挖掉前景(A)，换成图B -> Mask在前景为1
#     """
#     # 1. 下采样到 14x14 (ViT Patch Size)
#     # 使用 'nearest' 保持 0/1 的二值特性，或者 'area' 后再阈值化
#     # 这里假设 mask_pub 已经是 float 格式
#     mask_14 = F.interpolate(gt_mask, size=(14, 14), mode='nearest')
    
#     # 2. 确保是二值的 (0 或 1)
#     # 有时候插值会产生中间值，强制二值化
#     mask_14 = (mask_14 > 0.5).float()

#     # 3. 根据模式决定 Mask 的逻辑
#     # Mask 定义: 1 代表被替换(混合区域), 0 代表保留原图
    
#     if mode == 'keep_foreground':
#         # 目标：保留前景(GT=1的地方设为0)，替换背景(GT=0的地方设为1)
#         mask = 1.0 - mask_14
#     elif mode == 'swap_foreground':
#         # 目标：挖掉前景(GT=1的地方设为1)，保留背景(GT=0的地方设为0)
#         mask = mask_14
#     else:
#         raise ValueError(f"Unknown mode: {mode}")

#     # 4. 计算 lambda (保留原图的比例)
#     # 1 - (Mask面积 / 总面积)
#     N = 14 * 14
#     actual_lam = 1 - (mask.sum(dim=(1,2,3)) / N).mean().item()
    
#     return mask, actual_lam

# # # ==========================================
# # # 4. 核心 Mixup / TokenMix 类
# # # ==========================================

# # class TokenMixer:
# #     def __init__(self, mixup_alpha=1.0, cutmix_alpha=1.0, prob=0.5, switch_prob=0.5,
# #                  num_classes=1000, label_smoothing=0.1, mask_type='block'):
# #         self.mixup_alpha = mixup_alpha
# #         self.cutmix_alpha = cutmix_alpha
# #         self.prob = prob
# #         self.switch_prob = switch_prob
# #         self.num_classes = num_classes
# #         self.label_smoothing = label_smoothing
# #         self.mask_type = mask_type

# #     def _get_params(self):
# #         """决定使用 Mixup 还是 CutMix (TokenMix)"""
# #         if np.random.rand() > self.prob:
# #             return 1.0, False
        
# #         use_cutmix = np.random.rand() < self.switch_prob
# #         alpha = self.cutmix_alpha if use_cutmix else self.mixup_alpha
# #         lam = np.random.beta(alpha, alpha)
# #         return lam, use_cutmix

# #     def __call__(self, x, dense_labels, hard_labels, cross_x=None, cross_dense_labels=None, cross_target=None):
# #         lam, use_cutmix = self._get_params()
# #         B = x.shape[0]
# #         device = x.device

# #         if cross_x is not None and cross_dense_labels is not None and cross_target is not None:
# #             min_bs = min(B, cross_x.size(0))

# #             x = x[:min_bs]
# #             dense_labels = dense_labels[:min_bs]
# #             hard_labels = hard_labels[:min_bs]
# #             B = min_bs

# #             x_source_b = cross_x[:min_bs]
# #             dense_labels_b = cross_dense_labels[:min_bs]
# #             target_source_b = cross_target[:min_bs]
# #         else:
# #             x_source_b = x.flip(0)
# #             dense_labels_b = dense_labels.flip(0)
# #             target_source_b = hard_labels.flip(0)

# #         if use_cutmix:
# #             # --- TokenMix 模式 ---
# #             # 1. 生成 Batch 独立的 Mask (B, 1, 14, 14)
# #             mask, actual_lam = generate_mask_batch(B, 0.5, device, self.mask_type)
            
# #             # 2. 图像插值 Mask 到 224x224
# #             mask_224 = F.interpolate(mask, size=(224, 224), mode='nearest')
            
# #             # 3. 混合图像 (避免原地操作)
# #             x_mixed = x * (1 - mask_224) + x_source_b * mask_224
            
# #             # 4. 混合 Label (空间加权)
# #             # y1: 原图在保留区域的得分, y2: 翻转图在遮盖区域的得分
# #             y1_score = (dense_labels * (1 - mask)).sum(dim=(2, 3))
# #             y2_score = (dense_labels_b * mask).sum(dim=(2, 3))
            
# #             target = y1_score + y2_score
# #             # 归一化，防止某些样本由于 Saliency Map 总值极低导致数值不稳定
# #             target = target / (target.sum(dim=1, keepdim=True) + 1e-8)
            
# #             return x_mixed, target, mask_224
# #         else:
# #             # --- 标准 Mixup 模式 ---
# #             x_mixed = x * lam + x_source_b * (1. - lam)
# #             y1 = one_hot(hard_labels, self.num_classes, device=device)
# #             y2 = one_hot(target_source_b, self.num_classes, device=device)
# #             target = y1 * lam + y2 * (1. - lam)
# #             return x_mixed, target, torch.zeros_like(x)


# # 测试用
# class TokenMixer:
#     def __init__(self, mixup_alpha=1.0, cutmix_alpha=1.0, prob=0.5, switch_prob=0.5,
#                  num_classes=1000, label_smoothing=0.1, 
#                  mask_type='gt_swap_foreground'): # <--- 修改默认或传入这个
#         self.mixup_alpha = mixup_alpha
#         self.cutmix_alpha = cutmix_alpha
#         self.prob = prob
#         self.switch_prob = switch_prob
#         self.num_classes = num_classes
#         self.label_smoothing = label_smoothing
#         self.mask_type = mask_type 

#     def _get_params(self):
#         if np.random.rand() > self.prob:
#             return 1.0, False
#         use_cutmix = np.random.rand() < self.switch_prob
#         alpha = self.cutmix_alpha if use_cutmix else self.mixup_alpha
#         lam = np.random.beta(alpha, alpha)
#         return lam, use_cutmix

#     # 【关键修改】增加 gt_mask 参数
#     def __call__(self, x, dense_labels, hard_labels, gt_mask=None, cross_x=None, cross_dense_labels=None):
#         lam, use_cutmix = self._get_params()
#         B = x.shape[0]
#         device = x.device

#         # 数据翻转 (Source B)
#         if cross_x is not None:
#              min_bs = min(B, cross_x.size(0))
#              x = x[:min_bs]
#              dense_labels = dense_labels[:min_bs]
#              hard_labels = hard_labels[:min_bs]
#              if gt_mask is not None: gt_mask = gt_mask[:min_bs] # 对齐
#              B = min_bs
#              x_source_b = cross_x[:min_bs]
#              dense_labels_b = cross_dense_labels[:min_bs]
#         else:
#             x_source_b = x.flip(0)
#             dense_labels_b = dense_labels.flip(0)

#         if use_cutmix:
#             # =========================================
#             # Mask 生成逻辑
#             # =========================================
#             if self.mask_type == 'gt_keep_foreground':
#                 # 模式1: 保留前景，混合背景 (推荐)
#                 if gt_mask is None: raise ValueError("mask_type='gt_...' requires gt_mask input")
#                 mask, actual_lam = generate_mask_from_gt(gt_mask, device, mode='keep_foreground')
                
#             elif self.mask_type == 'gt_swap_foreground':
#                 # 模式2: 挖掉前景，换成图B (压力测试用)
#                 if gt_mask is None: raise ValueError("mask_type='gt_...' requires gt_mask input")
#                 mask, actual_lam = generate_mask_from_gt(gt_mask, device, mode='swap_foreground')

#             else:
#                 # 默认 block / random
#                 mask, actual_lam = generate_mask_batch(B, lam, device, self.mask_type)
            
#             # =========================================
#             # 混合操作
#             # =========================================
            
#             # 插值 mask 到 224x224
#             mask_224 = F.interpolate(mask, size=(224, 224), mode='nearest')
            
#             # 混合图像
#             x_mixed = x * (1 - mask_224) + x_source_b * mask_224
            
#             # 混合标签 (依然使用 Dense Label 计算能量，这是 TokenMix 的精髓)
#             # 即使 Mask 是由 GT 生成的，计算标签权重时最好还是用 Teacher 的注意力，
#             # 因为 GT Mask 只是 0/1，无法反映前景内部哪里更重要。
#             y1_score = (dense_labels * (1 - mask)).sum(dim=(2, 3))
#             y2_score = (dense_labels_b * mask).sum(dim=(2, 3))
            
#             target = y1_score + y2_score
#             target = target / (target.sum(dim=1, keepdim=True) + 1e-8)
            
#             return x_mixed, target, mask_224
#         else:
#             # Mixup
#             x_mixed = x * lam + x_source_b * (1. - lam)
#             y1 = one_hot(hard_labels, self.num_classes, device=device)
#             target_source_b = hard_labels.flip(0) 
#             y2 = one_hot(target_source_b, self.num_classes, device=device)
#             target = y1 * lam + y2 * (1. - lam)
#             return x_mixed, target, torch.zeros_like(x)

# from matplotlib import pyplot as plt

# def denormalize(tensor):
#     """还原归一化后的图像"""
#     mean = torch.tensor([0.485, 0.456, 0.406]).to(tensor.device).view(3, 1, 1)
#     std = torch.tensor([0.229, 0.224, 0.225]).to(tensor.device).view(3, 1, 1)
#     t = tensor.clone() * std + mean
#     return t.clamp(0, 1).permute(1, 2, 0).cpu().numpy()

# def visualize_tokenmix_complete(images, mixed_images, masks, saliency_maps, mixed_targets, original_labels, num_samples=4):
#     """
#     输出 A, B, 显著性图, mask 和 混合后的图像，并标注标签
#     """
#     # 图像 B 是 A 的翻转
#     images_b = images.flip(0)
#     labels_b = original_labels.flip(0)
    
#     fig, axes = plt.subplots(num_samples, 5, figsize=(20, 4 * num_samples))
    
#     for i in range(num_samples):
#         # --- Column 1: Original Image A ---
#         axes[i, 0].imshow(denormalize(images[i]))
#         axes[i, 0].set_title(f"Original A\nLabel: {original_labels[i].item()}", fontsize=12)
#         axes[i, 0].axis('off')

#         # --- Column 2: Teacher Saliency Map ---
#         # 融合所有类别的显著性，展示 Teacher 关注的区域
#         saliency = saliency_maps[i].sum(dim=0).cpu().numpy()
#         axes[i, 1].imshow(saliency, cmap='jet')
#         axes[i, 1].set_title("Saliency (Teacher)", fontsize=12)
#         axes[i, 1].axis('off')

#         # --- Column 3: The Mask ---
#         # 黑色=保留A，白色=填充B
#         mask_vis = masks[i, 0].cpu().numpy()
#         axes[i, 2].imshow(mask_vis, cmap='gray')
#         axes[i, 2].set_title("Mixing Mask\n(Black:A, White:B)", fontsize=12)
#         axes[i, 2].axis('off')

#         # --- Column 4: Source Image B ---
#         axes[i, 3].imshow(denormalize(images_b[i]))
#         axes[i, 3].set_title(f"Source B\nLabel: {labels_b[i].item()}", fontsize=12)
#         axes[i, 3].axis('off')

#         # --- Column 5: TokenMix Result ---
#         axes[i, 4].imshow(denormalize(mixed_images[i]))
#         # 提取软标签中权重最高的两个类别
#         top2_vals, top2_idx = torch.topk(mixed_targets[i], k=2)
#         res_label = f"L{top2_idx[0].item()}: {top2_vals[0].item():.2f}\nL{top2_idx[1].item()}: {top2_vals[1].item():.2f}"
#         axes[i, 4].set_title(f"Mixed Result\n{res_label}", fontsize=12, color='red')
#         axes[i, 4].axis('off')

#     plt.tight_layout()
#     plt.savefig("tokenmix_full_visualization.png")
#     print("\n[成功] 完整流程图已保存至: tokenmix_full_visualization.png")
#     plt.show()


import math
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from related_code.models import create_model
# from data_loader import get_data_loaders
# from transformers import ViTForImageClassification, ViTConfig


def one_hot(x, num_classes, on_value=1., off_value=0., device='cuda'):
    x = x.long().view(-1, 1)
    return torch.full((x.size()[0], num_classes), off_value, device=device).scatter_(1, x, on_value)

def mixup_target(target, num_classes, lam=1., smoothing=0.0, device='cuda'):
    off_value = smoothing / num_classes
    on_value = 1. - smoothing + off_value
    y1 = one_hot(target, num_classes, on_value=on_value, off_value=off_value, device=device)
    y2 = one_hot(target.flip(0), num_classes, on_value=on_value, off_value=off_value, device=device)
    return y1 * lam + y2 * (1. - lam)

# ==========================================
# 2. 改进的 Mask 生成 (支持 Batch 独立生成)
# ==========================================

def generate_single_block_mask(lam, device, mask_token_num_start=1):
    """生成单张图的 Block Mask (14x14)"""
    width, height = 14, 14
    mask = np.zeros(shape=(height, width), dtype=np.float32)
    mask_ratio = 1 - lam
    num_masking_patches = min(width * height, int(mask_ratio * width * height) + mask_token_num_start)

    mask_count = 0
    while mask_count < num_masking_patches:
        target_area = random.uniform(2, 10) # 每次涂抹的块大小
        aspect_ratio = random.uniform(0.3, 3.3)
        h = int(round(math.sqrt(target_area * aspect_ratio)))
        w = int(round(math.sqrt(target_area / aspect_ratio)))
        if w < width and h < height:
            top = random.randint(0, height - h)
            left = random.randint(0, width - w)
            for i in range(top, top + h):
                for j in range(left, left + w):
                    if mask[i, j] == 0 and mask_count < num_masking_patches:
                        mask[i, j] = 1
                        mask_count += 1
        if target_area > num_masking_patches - mask_count: break # 防止死循环

    mask = torch.from_numpy(mask).to(device).unsqueeze(0).unsqueeze(0) # (1, 1, 14, 14)
    return mask, 1 - (mask_count / (width * height))

def generate_mask_batch(batch_size, lam, device, mask_type='block', mask_token_num_start=14):
    """为整个 Batch 生成独立的 Mask"""
    masks = []
    lams = []
    for _ in range(batch_size):
        if mask_type == 'block':
            m, l = generate_single_block_mask(lam, device, mask_token_num_start)
        else: # random
            m_flat = torch.zeros(196, device=device)
            idx = torch.randperm(196, device=device)[:int(196 * (1-lam))]
            m_flat[idx] = 1
            m = m_flat.view(1, 1, 14, 14)
            l = 1 - (len(idx)/196)
        masks.append(m)
        lams.append(l)
    return torch.cat(masks, dim=0), np.mean(lams)

# ==========================================
# 3. 改进的 Dense Label 生成 (多层注意力融合)
# ==========================================

@torch.no_grad()
def generate_dense_labels(teacher_model, images, labels, num_classes=4, top_k_layers=3):
    """
    利用 Teacher 模型融合最后 K 层注意力生成更稳健的 Dense Labels
    """
    teacher_model.eval()
    # images: (B, 3, 224, 224)
    outputs = teacher_model.vit(pixel_values=images, output_attentions=True)

    # 1. 融合最后 K 层的 Attention Maps
    # attentions 是一元组，每层形状 (B, Heads, 197, 197), 197 = 1 (CLS) + 196 (Patch)
    relevant_attentions = outputs.attentions[-top_k_layers:]
    stack_attn = torch.stack(relevant_attentions) # (K, B, Heads, 197, 197)

    # 2. 对层维度和头维度取平均
    combined_attn = stack_attn.mean(dim=0).mean(dim=1) # (B, 197, 197)

    # 3. 提取 CLS Token 对所有 Patch 的注意力 (索引 0 为 CLS)
    cls_attn = combined_attn[:, 0, 1:] # (B, 196)

    # 4. 归一化 Saliency Map (Min-Max)
    B = cls_attn.shape[0]
    attn_min = cls_attn.min(dim=1, keepdim=True)[0]
    attn_max = cls_attn.max(dim=1, keepdim=True)[0]
    cls_attn = (cls_attn - attn_min) / (attn_max - attn_min + 1e-6)

    # 5. 生成空间热力图 (14x14)
    saliency_map = cls_attn.view(B, 1, 14, 14)

    # 6. 广播标签: (B, C, 1, 1) * (B, 1, 14, 14) -> (B, C, 14, 14)
    labels_onehot = F.one_hot(labels, num_classes=num_classes).float().view(B, num_classes, 1, 1)
    dense_labels = labels_onehot * saliency_map

    return dense_labels

# ==========================================
# 4. 核心 Mixup / TokenMix 类
# ==========================================

class TokenMixer:
    def __init__(self, mixup_alpha=1.0, cutmix_alpha=1.0, prob=0.5, switch_prob=0.5,
                 num_classes=1000, label_smoothing=0.1, mask_type='block'):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.switch_prob = switch_prob
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.mask_type = mask_type

    def _get_params(self):
        """决定使用 Mixup 还是 CutMix (TokenMix)"""
        if np.random.rand() > self.prob:
            return 1.0, False

        use_cutmix = np.random.rand() < self.switch_prob
        alpha = self.cutmix_alpha if use_cutmix else self.mixup_alpha
        lam = np.random.beta(alpha, alpha)
        return lam, use_cutmix

    def __call__(self, x, dense_labels, hard_labels, cross_x=None, cross_dense_labels=None, cross_target=None):
        lam, use_cutmix = self._get_params()
        B = x.shape[0]
        device = x.device

        if cross_x is not None and cross_dense_labels is not None and cross_target is not None:
            min_bs = min(B, cross_x.size(0))

            x = x[:min_bs]
            dense_labels = dense_labels[:min_bs]
            hard_labels = hard_labels[:min_bs]
            B = min_bs

            x_source_b = cross_x[:min_bs]
            dense_labels_b = cross_dense_labels[:min_bs]
            target_source_b = cross_target[:min_bs]
        else:
            x_source_b = x.flip(0)
            dense_labels_b = dense_labels.flip(0)
            target_source_b = hard_labels.flip(0)

        if use_cutmix:
            # --- TokenMix 模式 ---
            # 1. 生成 Batch 独立的 Mask (B, 1, 14, 14)
            mask, actual_lam = generate_mask_batch(B, 0.5, device, self.mask_type)

            # 2. 图像插值 Mask 到 224x224
            mask_224 = F.interpolate(mask, size=(224, 224), mode='nearest')

            # 3. 混合图像 (避免原地操作)
            x_mixed = x * (1 - mask_224) + x_source_b * mask_224

            # 4. 混合 Label (空间加权)
            # y1: 原图在保留区域的得分, y2: 翻转图在遮盖区域的得分
            y1_score = (dense_labels * (1 - mask)).sum(dim=(2, 3))
            y2_score = (dense_labels_b * mask).sum(dim=(2, 3))

            target = y1_score + y2_score
            # 归一化，防止某些样本由于 Saliency Map 总值极低导致数值不稳定
            target = target / (target.sum(dim=1, keepdim=True) + 1e-8)

            return x_mixed, target, mask_224
        else:
            # --- 标准 Mixup 模式 ---
            x_mixed = x * lam + x_source_b * (1. - lam)
            y1 = one_hot(hard_labels, self.num_classes, device=device)
            y2 = one_hot(target_source_b, self.num_classes, device=device)
            target = y1 * lam + y2 * (1. - lam)
            return x_mixed, target, torch.zeros_like(x)


from matplotlib import pyplot as plt

def denormalize(tensor):
    """还原归一化后的图像"""
    mean = torch.tensor([0.485, 0.456, 0.406]).to(tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).to(tensor.device).view(3, 1, 1)
    t = tensor.clone() * std + mean
    return t.clamp(0, 1).permute(1, 2, 0).cpu().numpy()

def visualize_tokenmix_complete(images, mixed_images, masks, saliency_maps, mixed_targets, original_labels, num_samples=4):
    """
    输出 A, B, 显著性图, mask 和 混合后的图像，并标注标签
    """
    # 图像 B 是 A 的翻转
    images_b = images.flip(0)
    labels_b = original_labels.flip(0)

    fig, axes = plt.subplots(num_samples, 5, figsize=(20, 4 * num_samples))

    for i in range(num_samples):
        # --- Column 1: Original Image A ---
        axes[i, 0].imshow(denormalize(images[i]))
        axes[i, 0].set_title(f"Original A\nLabel: {original_labels[i].item()}", fontsize=12)
        axes[i, 0].axis('off')

        # --- Column 2: Teacher Saliency Map ---
        # 融合所有类别的显著性，展示 Teacher 关注的区域
        saliency = saliency_maps[i].sum(dim=0).cpu().numpy()
        axes[i, 1].imshow(saliency, cmap='jet')
        axes[i, 1].set_title("Saliency (Teacher)", fontsize=12)
        axes[i, 1].axis('off')

        # --- Column 3: The Mask ---
        # 黑色=保留A，白色=填充B
        mask_vis = masks[i, 0].cpu().numpy()
        axes[i, 2].imshow(mask_vis, cmap='gray')
        axes[i, 2].set_title("Mixing Mask\n(Black:A, White:B)", fontsize=12)
        axes[i, 2].axis('off')

        # --- Column 4: Source Image B ---
        axes[i, 3].imshow(denormalize(images_b[i]))
        axes[i, 3].set_title(f"Source B\nLabel: {labels_b[i].item()}", fontsize=12)
        axes[i, 3].axis('off')

        # --- Column 5: TokenMix Result ---
        axes[i, 4].imshow(denormalize(mixed_images[i]))
        # 提取软标签中权重最高的两个类别
        top2_vals, top2_idx = torch.topk(mixed_targets[i], k=2)
        res_label = f"L{top2_idx[0].item()}: {top2_vals[0].item():.2f}\nL{top2_idx[1].item()}: {top2_vals[1].item():.2f}"
        axes[i, 4].set_title(f"Mixed Result\n{res_label}", fontsize=12, color='red')
        axes[i, 4].axis('off')

    plt.tight_layout()
    plt.savefig("tokenmix_full_visualization.png")
    print("\n[成功] 完整流程图已保存至: tokenmix_full_visualization.png")
    plt.show()