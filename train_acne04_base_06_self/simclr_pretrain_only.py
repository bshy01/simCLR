# simclr_pretrain_only.py
import os
import random
import time

import cv2
import numpy as np
from typing import Tuple

import albumentations as A
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import models as models_tv
from tqdm import tqdm

# ✅ 당신의 원본 데이터셋 (라벨은 여기선 사용하지 않음)
from dataset.datasets_base import DatasetAcne04Class

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# ----------------------------
# Albumentations 증강 (버전 1.x / 2.x 모두 호환)
# ----------------------------
def build_simclr_albu_transform(height: int = 224, width: int = 224):
    import albumentations as A
    _scale = (0.6, 1.0)
    _ratio = (0.8, 1.25)
    try:
        rrc = A.RandomResizedCrop(size=(height, width), scale=_scale, ratio=_ratio, p=1.0)
    except Exception:
        rrc = A.RandomResizedCrop(height=height, width=width, scale=_scale, ratio=_ratio, p=1.0)

    return A.Compose([
        A.ToFloat(max_value=255.0),  # ⬅️ 맨 앞으로 이동 (float32 보장)
        rrc,
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02, p=0.8),
        A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0.1, 2.0), p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])



# ----------------------------
# SimCLR용 데이터셋 래퍼
#   - 원본 DatasetAcne04Class에서 (CHW) 이미지를 받아
#   - HWC로 되돌린 후, 같은 이미지에 증강을 두 번 적용
#   - (2, C, H, W) 텐서 반환
# ----------------------------
class SimCLRDataset(Dataset):
    def __init__(self, base_dataset: Dataset, albu_transform):
        """
        base_dataset: DatasetAcne04Class 인스턴스 (transforms=None로 주는 걸 권장)
        albu_transform: Albumentations Compose (HWC numpy 입력)
        """
        self.base = base_dataset
        self.albu = albu_transform

    @staticmethod
    def _chw_to_hwc(img):
        # img: numpy (C,H,W) or (H,W) or (H,W,C)
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        img = np.array(img)

        if img.ndim == 3 and img.shape[0] in (1, 3, 4):  # (C,H,W)
            img = np.transpose(img, (1, 2, 0))  # (H,W,C)
        elif img.ndim == 2:  # (H,W)
            img = img[..., None]  # (H,W,1)
        # else: 이미 (H,W,C)

        # 채널 보정
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=2)  # 그레이스케일 → 3채널
        elif img.shape[-1] >= 3:
            img = img[..., :3]  # RGBA → RGB
        return img

    @staticmethod
    def _to_chw_tensor(np_img):
        # np_img: (H,W,C) float → torch (C,H,W) float
        if np_img.ndim == 2:
            np_img = np.expand_dims(np_img, axis=2)
            np_img = np.repeat(np_img, 3, axis=2)
        elif np_img.ndim == 3 and np_img.shape[2] == 1:
            np_img = np.repeat(np_img, 3, axis=2)
        elif np_img.ndim == 3 and np_img.shape[2] >= 3:
            np_img = np_img[:, :, :3]

        tensor = torch.from_numpy(np_img).permute(2, 0, 1).contiguous().float()
        return tensor

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, _ = self.base[idx]              # 보통 (C,H,W)
        img_hwc = self._chw_to_hwc(img)      # (H,W,C)

        # 🔧 dtype 강제: OpenCV가 싫어하는 float64 방지
        if img_hwc.dtype != np.uint8:
            m = float(np.max(img_hwc)) if img_hwc.size else 1.0
            if m <= 1.5:
                img_hwc = (np.clip(img_hwc, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                img_hwc = np.clip(img_hwc, 0.0, 255.0).astype(np.uint8)

        q = self.albu(image=img_hwc)["image"]   # numpy(H,W,C), 이제 float32로 변환됨(ToFloat)
        k = self.albu(image=img_hwc)["image"]

        q = self._to_chw_tensor(q)              # (C,H,W) float
        k = self._to_chw_tensor(k)
        return torch.stack([q, k], dim=0)       # (2,C,H,W)


# ----------------------------
# 모델 (Encoder + Projection Head)
# ----------------------------
class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 2048, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=True)
        )

    def forward(self, x):
        return self.net(x)


class SimCLR(nn.Module):
    def __init__(self, backbone_name: str = "resnet50", proj_hidden: int = 2048, proj_out: int = 128):
        super().__init__()
        assert backbone_name in ["resnet18", "resnet34", "resnet50", "resnet101"]

        if backbone_name == "resnet18":
            encoder = models_tv.resnet18(weights=models_tv.ResNet18_Weights.IMAGENET1K_V1)
        elif backbone_name == "resnet34":
            encoder = models_tv.resnet34(weights=models_tv.ResNet34_Weights.IMAGENET1K_V1)
        elif backbone_name == "resnet50":
            encoder = models_tv.resnet50(weights=models_tv.ResNet50_Weights.IMAGENET1K_V1)
        else:
            encoder = models_tv.resnet101(weights=models_tv.ResNet101_Weights.IMAGENET1K_V1)

        feat_dim = encoder.fc.in_features
        encoder.fc = nn.Identity()
        self.encoder = encoder
        self.projector = ProjectionHead(in_dim=feat_dim, hidden_dim=proj_hidden, out_dim=proj_out)

    def forward(self, x):
        h = self.encoder(x)               # (N, feat_dim)
        z = self.projector(h)             # (N, proj_out)
        z = F.normalize(z, dim=1)
        return h, z


# ----------------------------
# InfoNCE (NT-Xent) Loss
# ----------------------------
def info_nce_loss(z: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """
    z: (2N, D) normalized; 0..N-1: view1, N..2N-1: view2
    """
    N = z.shape[0] // 2
    z1, z2 = z[:N], z[N:]

    logits = (z1 @ z2.T) / temperature  # (N, N)
    labels = torch.arange(N, device=z.device, dtype=torch.long)

    loss_12 = F.cross_entropy(logits, labels)
    loss_21 = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss_12 + loss_21)


# ----------------------------
# Collate: (B, 2, C, H, W) → (2B, C, H, W)
# ----------------------------
def collate_fn(batch):
    # batch: list of (2, C, H, W) tensors
    x = torch.stack(batch, dim=0)  # (B, 2, C, H, W)
    x = x.view(-1, x.shape[2], x.shape[3], x.shape[4])  # (2B, C, H, W)
    return x


# ----------------------------
# Train Loop
# ----------------------------
def train_simclr(
    path_tr_src: str,
    save_dir: str = "save_pretrain/SimCLR_Acne04",
    log_dir: str = "logs_simclr/Acne04",
    backbone: str = "resnet50",
    epochs: int = 200,
    batch_size: int = 64,
    accumulation_steps: int = 1,
    lr: float = 3e-4,
    temperature: float = 0.1,
    img_h: int = 224,
    img_w: int = 224,
    seed: int = 3407,
    num_workers: int = 0,
):
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device :', device)
    if torch.cuda.is_available():
        print('Current cuda device:', torch.cuda.current_device())
        print('Count of using GPUs:', torch.cuda.device_count())

    # ✅ 원본 데이터셋: transforms=None 로 받아 CHW로 나오게 두고,
    #    (HWC 변환 및 증강 두 번)은 SimCLRDataset에서 수행
    base_dataset = DatasetAcne04Class(
        path_src=path_tr_src,
        transforms=None,
        color=cv2.IMREAD_COLOR_BGR,
        colorspace=None
    )
    albu_aug = build_simclr_albu_transform(img_h, img_w)
    tr_dataset = SimCLRDataset(base_dataset, albu_aug)

    tr_loader = DataLoader(
        tr_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn
    )

    # Model / Optim / Scheduler
    model = SimCLR(backbone_name=backbone, proj_hidden=2048, proj_out=128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Train
    global_step = 0
    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        loss_m = []
        optimizer.zero_grad()

        pbar = tqdm(enumerate(tr_loader), desc=f"[Epoch {epoch+1}/{epochs}]", total=len(tr_loader))
        for i, x in pbar:
            x = x.to(device, non_blocking=True)      # (2B, C, H, W)

            _, z = model(x)                          # (2B, D)
            loss = info_nce_loss(z, temperature=temperature)

            if accumulation_steps > 1:
                loss = loss / accumulation_steps

            loss.backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(tr_loader):
                optimizer.step()
                optimizer.zero_grad()

            lv = loss.item()
            if accumulation_steps > 1:
                lv = lv * accumulation_steps

            loss_m.append(lv)
            writer.add_scalar("train/loss", lv, global_step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], global_step)
            pbar.set_postfix(loss=f"{lv:.4f}")
            global_step += 1

        epoch_loss = float(np.mean(loss_m))
        print(f"Epoch {epoch+1}: loss={epoch_loss:.6f}, lr={optimizer.param_groups[0]['lr']:.6e}")
        scheduler.step()

        # Save best (by loss)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            ckpt_path = os.path.join(save_dir, f"simclr_{backbone}_best.pth")
            torch.save({
                "epoch": epoch,
                "backbone": backbone,
                "state_dict": model.state_dict(),                  # encoder+projector
                "encoder_state_dict": model.encoder.state_dict(),  # 👉 파인튜닝 시 주로 사용
                "proj_state_dict": model.projector.state_dict(),
                "loss": epoch_loss,
            }, ckpt_path)
            print(f"[Saved] {ckpt_path} (loss={best_loss:.6f})")

        # Save last each epoch
        ckpt_last = os.path.join(save_dir, f"simclr_{backbone}_last.pth")
        torch.save({
            "epoch": epoch,
            "backbone": backbone,
            "state_dict": model.state_dict(),
            "encoder_state_dict": model.encoder.state_dict(),
            "proj_state_dict": model.projector.state_dict(),
            "loss": epoch_loss,
        }, ckpt_last)

    writer.flush()
    writer.close()
    print("Pretraining done.")


# ----------------------------
# Entry
# ----------------------------
if __name__ == "__main__":
    # ✅ 반드시 Train 폴더만 사용 (Test/Val 금지: 누수 방지)
    path_tr_src = r"/shared/data/ACNE04_Total/Semi-supervised/0.40_Tr_labeled"

    train_simclr(
        path_tr_src=path_tr_src,
        save_dir="save_pretrain/SimCLR_Acne04_0.40",
        log_dir="logs_simclr/Acne04",
        backbone="resnet50",       # resnet18/34/50/101 가능
        epochs=50,
        batch_size=128,             # RTX 4090에서 시도해볼만한 배치
        accumulation_steps=8,      # 128 * 8 = 1024 effective batch size
        lr=3e-4,
        temperature=0.1,
        img_h=224,
        img_w=224,
        seed=3407,
        num_workers=4,             # PyCharm 디버깅 시 0이 안전, 리눅스 학습은 4~8 권장
    )
