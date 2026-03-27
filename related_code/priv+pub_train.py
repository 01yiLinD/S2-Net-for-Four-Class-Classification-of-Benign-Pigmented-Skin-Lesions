import copy
import os
import torch
import random
import numpy as np
import sys
import torch.nn as nn
import torch.nn.functional as F

from itertools import cycle 
from torch.utils.tensorboard import SummaryWriter
from utils.data_loader import get_data_loaders 
from models import create_model   
from utils.TokenMix import TokenMixer, generate_dense_labels
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def validate_domain(model, loader, criterion, device, num_classes):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    total_loss = 0.0

    with torch.no_grad():
        for imgs, labels, _ in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)

            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)

    f1 = f1_score(all_labels, all_preds, average='macro')
    
    try:
        auroc = roc_auc_score(all_labels, all_probs, multi_class='ovr', labels=range(num_classes))
    except ValueError:
        auroc = 0.0
        print("Warning: AUROC calculation failed (possibly missing classes in validation set).")

    return {
        'loss': avg_loss,
        'acc': acc,
        'f1': f1,
        'auroc': auroc,
        'combined': f1 + auroc
    }


def train_model(
    model_name="resnet",
    root_dir="ISIC-pifujing",
    save_dir="Experiment/baseline",
    batch_size=64,
    lr=1e-4,
    num_epochs=100,
    optimizer_type="adamw",
    device=None,
):
    os.makedirs(save_dir, exist_ok=True)

    log_file_path = os.path.join(save_dir, "training_log.txt")
    sys.stdout = Logger(log_file_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    writer = SummaryWriter(log_dir=os.path.join(save_dir, "logs")) 

    # 1. Load Data
    BATCH_SIZE = batch_size
    PUBLIC_PATH = root_dir 
    PRIVATE_PATH = "cropped_lesions_padded"

    (pub_train, pub_val, _,
     priv_train, priv_val, _) = get_data_loaders(
         PUBLIC_PATH, PRIVATE_PATH, batch_size=BATCH_SIZE
    )

    #------------SemanticMix Setup------------#
    # 2. Initialize Teacher Model

    print("Initializing teacher model...")
    pub_teacher_model = create_model(model_type="transformer", pretrained=False).to(device)
    
    if hasattr(pub_teacher_model, 'vit'):
        pub_teacher_model.vit.set_attn_implementation("eager")
    elif hasattr(pub_teacher_model, 'set_attn_implementation'):
        pub_teacher_model.set_attn_implementation("eager")

    pub_teacher_ckpt = "MyModel_Public/Transformer/augment+sampler_lr=1e-4/best_transformer_checkpoint.pth"
    pub_teacher_model.load_state_dict(torch.load(pub_teacher_ckpt)['model_state_dict'])
    pub_teacher_model.eval()

    for param in pub_teacher_model.parameters():
        param.requires_grad = False

    # 3. Initialize TokenMixer
    mixer = TokenMixer(
        mixup_alpha=0.8,
        cutmix_alpha=1.0,
        prob=1.0,
        switch_prob=0.5,
        num_classes=4,
        mask_type='block'
    )
    #------------------------------------------#
    
    # model = create_model(model_type="resnetmixstyle", pretrained=True).to(device)
    model = create_model(model_type=model_name, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()

    if optimizer_type.lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    elif optimizer_type.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    warmup_epochs = 10
    scheduler_warmup = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_epochs
    )

    scheduler_cosine = CosineAnnealingLR(
        optimizer,
        T_max=num_epochs - warmup_epochs,
        eta_min=1e-6
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[scheduler_warmup, scheduler_cosine],
        milestones=[warmup_epochs]
    )


    best_combined_score = 0.0
    
    def to_one_hot(labels, num_classes):
        return F.one_hot(labels, num_classes=num_classes).float()
        

    for epoch in range(num_epochs):
        if len(pub_train) > len(priv_train):
            train_loader = zip(pub_train, cycle(priv_train))
            num_batches = len(pub_train)
        else:
            train_loader = zip(cycle(pub_train), priv_train)
            num_batches = len(priv_train)

        print(f"Epoch {epoch+1}/{num_epochs}")

        model.train()
        total_train_loss = 0.0
        total_samples_seen = 0

        for batch_idx, ((img_pub, label_pub, mask_pub), (img_priv, label_priv, mask_priv)) in enumerate(train_loader):
            min_b = min(img_pub.size(0), img_priv.size(0))
            img_pub, label_pub, mask_pub = img_pub[:min_b].to(device), label_pub[:min_b].to(device).long(), mask_pub[:min_b].to(device).float()
            img_priv, label_priv, mask_priv = img_priv[:min_b].to(device), label_priv[:min_b].to(device).long(), mask_priv[:min_b].to(device).float()
       
            #---------------------------------------Cross domain TokenMix----------------------------------------------
            # if random.random() < 0.5:
            #     with torch.no_grad():  
            #         dense_labels_pub = generate_dense_labels(pub_teacher_model, img_pub, label_pub, num_classes=4, top_k_layers=3).to(device)
            #         dense_labels_priv = generate_dense_labels(pub_teacher_model, img_priv, label_priv, num_classes=4, top_k_layers=3).to(device)

            #         dense_labels_pub = dense_labels_pub * mask_pub
            #         dense_labels_priv = dense_labels_priv * mask_priv

            #     img_mix_1, target_mix_1, _ = mixer(
            #         x=img_pub,
            #         dense_labels=dense_labels_pub,
            #         hard_labels=label_pub,
            #         cross_x=img_priv,
            #         cross_dense_labels=dense_labels_priv,
            #         cross_target=label_priv
            #     )

            #     img_mix_2, target_mix_2, _ = mixer(
            #         x=img_priv, 
            #         dense_labels=dense_labels_priv, 
            #         hard_labels=label_priv,
            #         cross_x=img_pub,
            #         cross_dense_labels=dense_labels_pub,
            #         cross_target=label_pub
            #     )

            #     imgs = torch.cat((img_mix_1, img_mix_2), dim=0)
            #     targets = torch.cat((target_mix_1, target_mix_2), dim=0)
            # else:
            #     imgs = torch.cat((img_pub, img_priv), dim=0)
            #     targets = torch.cat((to_one_hot(label_pub, 4), to_one_hot(label_priv, 4)), dim=0)
            #------------------------------------------------------------------------------------------

            if random.random() < 0.5:
                with torch.no_grad():  
                    dense_labels_pub = generate_dense_labels(pub_teacher_model, img_pub, label_pub, num_classes=4, top_k_layers=3).to(device) 
                    dense_labels_priv = generate_dense_labels(pub_teacher_model, img_priv, label_priv, num_classes=4, top_k_layers=3).to(device)

                    # print(f"dense_labels_pub: {dense_labels_pub.shape}, dense_label_priv: {dense_labels_priv.shape}")
                    # print(f"mask_pub: {mask_pub.shape}, mask_priv: {mask_priv.shape}")

                    dense_labels_pub = dense_labels_pub * mask_pub
                    dense_labels_priv = dense_labels_priv * mask_priv

                img_pub, label_pub, _ = mixer(img_pub, dense_labels_pub, label_pub)
                img_priv, label_priv, _ = mixer(img_priv, dense_labels_priv, label_priv)

                # print(f"label_pub: {label_pub.shape}")
                # print(f"label_priv: {label_priv.shape}")

                imgs = torch.cat((img_pub, img_priv), dim=0)
                targets = torch.cat((label_pub, label_priv), dim=0)
            else:
                imgs = torch.cat((img_pub, img_priv), dim=0)
                label_pub = to_one_hot(label_pub, num_classes=4)
                label_priv = to_one_hot(label_priv, num_classes=4)
                targets = torch.cat((label_pub, label_priv), dim=0)

            # ------------------------------------------------------------------------------------------

            outputs = model(imgs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_batch_size = imgs.size(0)
            total_train_loss += loss.item() * current_batch_size
            total_samples_seen += current_batch_size

            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{num_batches} | Loss: {loss.item():.4f}", end='\r')

        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("LR", current_lr, epoch + 1)

        avg_train_loss = total_train_loss / total_samples_seen
        print(f"Epoch {epoch+1}/{num_epochs} | Average Training Loss: {avg_train_loss:.4f}")
        writer.add_scalar("Loss/Train", avg_train_loss, epoch+1)
    
        pub_metrics = validate_domain(model, pub_val, criterion, device, num_classes=4)
        priv_metrics = validate_domain(model, priv_val, criterion, device, num_classes=4)

        # current_combined = (pub_metrics["combined"] + priv_metrics["combined"]) / 2.0
        current_combined = priv_metrics["combined"]

        print(f"Public  - Loss: {pub_metrics['loss']:.4f}, Acc: {pub_metrics['acc']:.4f}, "
          f"F1: {pub_metrics['f1']:.4f}, AUROC: {pub_metrics['auroc']:.4f}, "
          f"Sum: {pub_metrics['combined']:.4f}")  

        print(f"Private - Loss: {priv_metrics['loss']:.4f}, Acc: {priv_metrics['acc']:.4f}, "
          f"F1: {priv_metrics['f1']:.4f}, AUROC: {priv_metrics['auroc']:.4f}, "
          f"Sum: {priv_metrics['combined']:.4f}")

        if current_combined > best_combined_score:
            best_combined_score = current_combined
            save_path = os.path.join(save_dir, 'best_mixstyle_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_score': best_combined_score,
                'pub_metrics': pub_metrics,
                'priv_metrics': priv_metrics
            }, save_path)
            print(f"--> Best model saved! (Combined Score: {best_combined_score:.4f})")

        writer.add_scalar("Loss/Pub_Val", pub_metrics['loss'], epoch + 1)
        writer.add_scalar("Loss/Priv_Val", priv_metrics['loss'], epoch + 1)
        writer.add_scalar("Acc/Pub", pub_metrics['acc'], epoch + 1)
        writer.add_scalar("Acc/Priv", priv_metrics['acc'], epoch + 1)
        writer.add_scalar("F1/Pub", pub_metrics['f1'], epoch + 1)  
        writer.add_scalar("F1/Priv", priv_metrics['f1'], epoch + 1)
        writer.add_scalar("AUROC/Pub", pub_metrics['auroc'], epoch + 1)
        writer.add_scalar("AUROC/Priv", priv_metrics['auroc'], epoch + 1)
        writer.add_scalar("Combined_Score/Pub", pub_metrics['combined'], epoch + 1)
        writer.add_scalar("Combined_Score/Priv", priv_metrics['combined'], epoch + 1)
        writer.add_scalar("Combined_Score/Total", current_combined, epoch + 1)


if __name__ == "__main__":
    set_seed(42)
    train_model(
        model_name="transformer",
        root_dir="ISIC-pifujing",
        save_dir="Experiment/resnet/S2-Net",
        batch_size=64,
        lr=1e-4,
        num_epochs=100,
        optimizer_type="adamw",
        device=None,
    )