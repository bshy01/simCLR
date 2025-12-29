import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import torch
from torch import nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import random
import torch.nn.functional as F
from torchvision import models as models_tv
import argparse

from dataset.datasets_base import DatasetAcne04Class

# This script is designed for linear evaluation of a pre-trained SimCLR model.

seed = 3407
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)

# =========================
# Model Components
# =========================
# This is a simplified SimCLR-like wrapper to load the pretrained model's state_dict.
# It only needs the encoder part for the linear evaluation.
class SimCLRWrapper(nn.Module):
    def __init__(self, base_model='resnet50', proj_out_dim=128):
        super().__init__()
        backbone = getattr(models_tv, base_model)(pretrained=False) # pretrained=False as we load our own weights
        feat_dim = backbone.fc.in_features
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        # The projection head is needed to match the keys in the state_dict
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, proj_out_dim)
        )

class LinearClassifier(nn.Module):
    def __init__(self, in_dim, num_class):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_class)

    def forward(self, x):
        return self.fc(x)

class FrozenBackbone(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        # Freeze the backbone
        for p in self.encoder.parameters():
            p.requires_grad = False

    def forward(self, x):
        h = self.encoder(x)
        h = torch.flatten(h, 1)
        return h

# =========================
# Data Utilities
# =========================
class EvalWrapper(torch.utils.data.Dataset):
    def __init__(self, base, aug):
        self.base = base
        self.aug = aug
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        img, label = self.base[idx]
        x = self.aug(image=to_hwc_rgb_uint8(img))['image']
        return x, label

def to_hwc_rgb_uint8(img):
    import numpy as np
    import torch

    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    if img.ndim == 3 and img.shape[0] in (1, 3, 4):
        img = np.transpose(img, (1, 2, 0))
    if img.ndim == 2:
        img = img[..., None]
    if img.shape[-1] == 4:
        img = img[..., :3]
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, None)
        maxv = float(img.max()) if img.size else 1.0
        if maxv <= 1.0:
            img = (img * 255.0).round().astype(np.uint8)
        else:
            img = img.round().astype(np.uint8)
    return img

def remap_resnet_keys(state_dict):
    """Remaps torchvision resnet keys to sequential model keys."""
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k
        if k.startswith('conv1'):
            new_k = k.replace('conv1', '0', 1)
        elif k.startswith('bn1'):
            new_k = k.replace('bn1', '1', 1)
        elif k.startswith('layer1'):
            new_k = k.replace('layer1', '4', 1)
        elif k.startswith('layer2'):
            new_k = k.replace('layer2', '5', 1)
        elif k.startswith('layer3'):
            new_k = k.replace('layer3', '6', 1)
        elif k.startswith('layer4'):
            new_k = k.replace('layer4', '7', 1)
        new_state_dict[new_k] = v
    return new_state_dict


def main():
    parser = argparse.ArgumentParser(description='Linear Evaluation of a Pre-trained SimCLR model based on the original paper protocol.')
    parser.add_argument('--model_path', type=str, default='train_acne04_base_06_self/simclr_pretrain_only/best.pth', help='Path to the pretrained SimCLR model')
    parser.add_argument('--num_epochs_lin', type=int, default=90, help='Epochs for linear evaluation (SimCLR default: 90)')
    parser.add_argument('--val_batch_size', type=int, default=16, help='Batch size for evaluation')
    parser.add_argument('--learning_rate_lin', type=float, default=0.1, help='Learning rate for linear eval (SGD)')
    parser.add_argument('--num_class', type=int, default=200, help='Number of classes for linear evaluation')
    parser.add_argument('--len_h', type=int, default=224, help='Image height')
    parser.add_argument('--len_w', type=int, default=224, help='Image width')
    parser.add_argument('--str_save', type=str, default='Acne04_Linear_Eval', help='Save directory name')
    parser.add_argument('--path_tr_src', type=str, default='/shared/data/ACNE04_Total/Cropped Faces/Tr/0.05_Tr_labeled', help='Training data path')
    parser.add_argument('--path_te_src', type=str, default='/shared/data/ACNE04_Total/Cropped Faces/Te', help='Test data path')
    args = parser.parse_args()
    args.path_save = f'save_model/{args.str_save}'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device:', device)
    if torch.cuda.is_available():
        print('Current cuda device:', torch.cuda.current_device())
        print('Count of using GPUs:', torch.cuda.device_count())

    # =========================
    # Data Augmentation and Loaders (SimCLR Protocol)
    # =========================
    # Augmentation for training the linear classifier
    train_aug = A.Compose([
        A.RandomResizedCrop(height=args.len_h, width=args.len_w),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    # Augmentation for evaluating the linear classifier
    test_aug = A.Compose([
        A.Resize(height=args.len_h, width=args.len_w),
        A.CenterCrop(height=args.len_h, width=args.len_w),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    base_tr = DatasetAcne04Class(path_src=args.path_tr_src, transforms=None, color=cv2.IMREAD_COLOR)
    base_te = DatasetAcne04Class(path_src=args.path_te_src, transforms=None, color=cv2.IMREAD_COLOR)
    tr_eval_dataset = EvalWrapper(base_tr, train_aug)
    te_eval_dataset = EvalWrapper(base_te, test_aug)
    tr_eval_loader = torch.utils.data.DataLoader(tr_eval_dataset, batch_size=args.val_batch_size, shuffle=True, num_workers=0, pin_memory=True)
    te_eval_loader = torch.utils.data.DataLoader(te_eval_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # =========================
    # Load Pre-trained Model
    # =========================
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Pretrained model not found at: {args.model_path}")

    print(f"Loading pretrained model from: {args.model_path}")
    # We need to instantiate the wrapper to load the state_dict
    pretrained_model = SimCLRWrapper(base_model='resnet50', proj_out_dim=128).to(device)
    
    # The saved model from simclr_pretrain_only.py might be just the encoder.
    # Let's check the state dict keys.
    ckpt = torch.load(args.model_path, map_location=device)
    
    # If 'state_dict' is a key, it's likely a checkpoint from a framework like PyTorch Lightning
    if 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']

    # Prepare encoder state dict
    encoder_state_dict = ckpt
    if any(k.startswith('encoder.') for k in ckpt.keys()):
        encoder_state_dict = {k.replace('encoder.', ''): v for k, v in ckpt.items() if k.startswith('encoder.')}
    elif any(k.startswith('module.encoder.') for k in ckpt.keys()): # Handle DataParallel
        encoder_state_dict = {k.replace('module.encoder.', ''): v for k, v in ckpt.items() if k.startswith('module.encoder.')}

    try:
        pretrained_model.encoder.load_state_dict(encoder_state_dict, strict=True)
    except RuntimeError as e:
        print("Failed to load encoder state_dict directly. Attempting to remap keys for Sequential model.")
        remapped_state_dict = remap_resnet_keys(encoder_state_dict)
        pretrained_model.encoder.load_state_dict(remapped_state_dict, strict=True)

    print("Pretrained model loaded successfully.")

    # =========================
    # Linear Evaluation (SimCLR Protocol)
    # =========================
    # The feature dimension is the input to the projection head.
    feat_dim = pretrained_model.proj[0].in_features
    
    backbone = FrozenBackbone(pretrained_model.encoder).to(device)
    classifier = LinearClassifier(in_dim=feat_dim, num_class=args.num_class).to(device)
    criterion_ce = nn.CrossEntropyLoss()
    # SimCLR paper uses LARS, but SGD with momentum is a common alternative for linear evaluation.
    optimizer_lin = torch.optim.SGD(classifier.parameters(), momentum=0.9, weight_decay=5e-4, lr=args.learning_rate_lin)
    scheduler_lin = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_lin, T_max=args.num_epochs_lin)

    writer_tr = SummaryWriter(log_dir=f'logs/{args.str_save}/linear_train')
    writer_te = SummaryWriter(log_dir=f'logs/{args.str_save}/linear_val')
    writer_acc = SummaryWriter(log_dir=f'logs/{args.str_save}/linear_acc')

    best_acc = 0.0
    for epoch in range(args.num_epochs_lin):
        # --- Train ---
        classifier.train()
        backbone.eval()
        train_losses = []
        for x, y in tqdm(tr_eval_loader, desc=f'[Linear Train] Epoch {epoch+1}/{args.num_epochs_lin}'):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.no_grad():
                feat = backbone(x)
            logits = classifier(feat)
            loss = criterion_ce(logits, y)
            optimizer_lin.zero_grad()
            loss.backward()
            optimizer_lin.step()
            train_losses.append(loss.item())
        scheduler_lin.step()
        tr_loss = float(np.mean(train_losses))
        writer_tr.add_scalar('loss', tr_loss, epoch)

        # --- Eval ---
        classifier.eval()
        val_losses, correct, total = [], 0, 0
        with torch.no_grad():
            for x, y in tqdm(te_eval_loader, desc=f'[Linear Eval] Epoch {epoch+1}/{args.num_epochs_lin}'):
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                feat = backbone(x)
                logits = classifier(feat)
                loss = criterion_ce(logits, y)
                val_losses.append(loss.item())
                pred = torch.argmax(logits, dim=1)
                total += y.size(0)
                correct += (pred == y).sum().item()

        acc = correct / total
        val_loss = float(np.mean(val_losses))
        print(f'[Linear Eval] epoch {epoch:03d} | tr_loss {tr_loss:.4f} | val_loss {val_loss:.4f} | acc {acc:.4f}')
        writer_te.add_scalar('loss', val_loss, epoch)
        writer_acc.add_scalar('acc', acc, epoch)

        if acc > best_acc:
            best_acc = acc
            os.makedirs(args.path_save, exist_ok=True)
            ckpt_path = os.path.join(args.path_save, f'linear_best_acc_{best_acc:.4f}.pth')
            torch.save({
                'backbone_state': backbone.encoder.state_dict(),
                'classifier_state': classifier.state_dict(),
                'acc': best_acc, 'epoch': epoch
            }, ckpt_path)
            print(f"Best model saved to {ckpt_path}")

    writer_tr.flush(); writer_te.flush(); writer_acc.flush()
    print(f'Linear evaluation finished. Best linear eval Acc: {best_acc:.4f}')

if __name__ == '__main__':
    main()
