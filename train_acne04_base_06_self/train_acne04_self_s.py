import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import random
import torch.nn.functional as F
from torchvision import models as models_tv

from dataset.datasets_base import DatasetAcne04Class

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

seed = 3407
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)

# =========================
# SimCLR 구성요소
# =========================
class ProjectionHead(nn.Module):
    """2-layer MLP: d -> 2048 -> 128 (ReLU)"""
    def __init__(self, in_dim, hidden_dim=2048, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)

def simclr_nt_xent(z1, z2, temperature=0.2):
    """
    z1, z2: (B, D) after projection, L2-normalized
    Returns NT-Xent loss for SimCLR.
    """
    B, D = z1.shape
    z = torch.cat([z1, z2], dim=0)                     # (2B, D)
    sim = torch.matmul(z, z.T) / temperature           # cosine sim since z normalized

    # mask to remove similarity with itself
    mask = torch.eye(2*B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, -9e15)

    # positives: for i in [0..B-1], pos of i is i+B; and for i in [B..2B-1], pos is i-B
    pos = torch.cat([torch.arange(B, 2*B), torch.arange(0, B)]).to(z.device)

    # log-softmax on rows
    log_prob = F.log_softmax(sim, dim=1)
    loss = -log_prob[torch.arange(2*B).to(z.device), pos].mean()
    return loss

class SimCLRWrapper(nn.Module):
    """
    ResNet50 backbone + proj head.
    forward(x) -> normalized projection
    """
    def __init__(self, base_model='resnet50', proj_out_dim=128):
        super().__init__()
        backbone = getattr(models_tv, base_model)(pretrained=True)
        feat_dim = backbone.fc.in_features
        # remove classifier head
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])  # up to avgpool -> (B, C, 1,1)
        self.proj = ProjectionHead(feat_dim, 2048, proj_out_dim)

    def forward(self, x):
        h = self.encoder(x)                      # (B, C, 1, 1)
        h = torch.flatten(h, 1)                  # (B, C)
        z = self.proj(h)                         # (B, D)
        z = F.normalize(z, dim=1)                # L2-norm
        return z

def to_hwc_rgb_uint8(img):
    import numpy as np
    import torch

    # torch.Tensor -> numpy
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()

    # CHW -> HWC
    if img.ndim == 3 and img.shape[0] in (1, 3, 4):
        img = np.transpose(img, (1, 2, 0))

    # grayscale 2D -> HWC(1)
    if img.ndim == 2:
        img = img[..., None]

    # 4채널이면 alpha 제거
    if img.shape[-1] == 4:
        img = img[..., :3]

    # 1채널이면 3채널로 복제
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)

    # dtype 정리 (0~1 float 또는 0~255 float -> uint8)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, None)
        maxv = float(img.max()) if img.size else 1.0
        if maxv <= 1.0:
            img = (img * 255.0).round().astype(np.uint8)
        else:
            img = img.round().astype(np.uint8)
    return img


# 두 뷰를 뽑아주는 래퍼 데이터셋
class SimCLRDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, aug1, aug2, bgr_to_rgb=False):
        self.base = base_dataset
        self.aug1 = aug1
        self.aug2 = aug2
        self.bgr_to_rgb = bgr_to_rgb  # cv2로 BGR 읽는 경우 True로

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]

        # Albumentations 전처리 호환 형태로 통일
        img = to_hwc_rgb_uint8(img)

        # (선택) BGR -> RGB
        if self.bgr_to_rgb:
            img = img[..., ::-1].copy()

        x1 = self.aug1(image=img)['image']
        x2 = self.aug2(image=img)['image']
        return x1, x2, label


# 선형평가용 분류기
class LinearClassifier(nn.Module):
    def __init__(self, in_dim, num_class):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_class)

    def forward(self, x):
        return self.fc(x)

# 백본에서 feature만 뽑는 모듈
class FrozenBackbone(nn.Module):
    def __init__(self, base_model='resnet50'):
        super().__init__()
        backbone = getattr(models_tv, base_model)(pretrained=False)
        feat_dim = backbone.fc.in_features
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.feat_dim = feat_dim
        for p in self.encoder.parameters():
            p.requires_grad = False

    def forward(self, x):
        h = self.encoder(x)
        h = torch.flatten(h, 1)
        return h

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device :', device)
    if torch.cuda.is_available():
        print('Current cuda device:', torch.cuda.current_device())
        print('Count of using GPUs:', torch.cuda.device_count())

    # =========================
    # 하이퍼파라미터
    # =========================
    num_epochs_ssl = 200          # 사전학습 epoch
    num_epochs_lin = 50           # 선형평가 epoch
    train_batch_size = 32        # SimCLR는 큰 배치가 유리
    val_batch_size = 16
    learning_rate_ssl = 3e-4
    learning_rate_lin = 1e-3
    temperature = 0.2
    num_class = 200

    len_h = 224
    len_w = 224
    color = cv2.IMREAD_COLOR
    colorspace = None

    str_save = 'Acne04_SimCLR_ResNet50'

    path_tr_src = '/shared/data/ACNE04_Total/Cropped Faces/Tr/0.05_Tr_labeled'
    path_te_src = '/shared/data/ACNE04_Total/Cropped Faces/Te'

    path_save = f'save_model/{str_save}'
    os.makedirs(path_save, exist_ok=True)

    # =========================
    # 증강 정의
    #   - SimCLR 권장: 강한 색변형/크롭/블러/수평뒤집기/그레이스케일
    # =========================
    normalize = A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
    simclr_aug1 = A.Compose([
        A.RandomResizedCrop(size=(len_h, len_w), scale=(0.2, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
        # A.ToGray(p=0.2),  # RGB 색정보가 중요하면 제거 권장
        A.GaussianBlur(blur_limit=(3, 5), p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    simclr_aug2 = A.Compose([
        A.RandomResizedCrop(size=(len_h, len_w), scale=(0.2, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
        # A.ToGray(p=0.2),
        A.GaussianBlur(blur_limit=(3, 5), p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    eval_aug = A.Compose([
        A.Resize(height=len_h, width=len_w),  # Resize는 그대로 height/width 유지
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    # =========================
    # 데이터셋/로더
    # =========================
    # base dataset은 (img, label) 반환한다고 가정
    base_tr = DatasetAcne04Class(path_src=path_tr_src, transforms=None, color=color, colorspace=colorspace)
    base_te = DatasetAcne04Class(path_src=path_te_src, transforms=None, color=color, colorspace=colorspace)

    # SimCLR용 래퍼(두 뷰)
    tr_ssl_dataset = SimCLRDataset(base_tr, simclr_aug1, simclr_aug2)
    tr_ssl_loader = torch.utils.data.DataLoader(tr_ssl_dataset, batch_size=train_batch_size,
                                                shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

    # 평가용(선형평가 단계)
    class EvalWrapper(torch.utils.data.Dataset):
        def __init__(self, base, aug):
            self.base = base
            self.aug = aug
        def __len__(self): return len(self.base)
        def __getitem__(self, idx):
            img, label = self.base[idx]
            x = self.aug(image=img)['image']
            return x, label

    tr_eval_dataset = EvalWrapper(base_tr, eval_aug)
    te_eval_dataset = EvalWrapper(base_te, eval_aug)
    tr_eval_loader = torch.utils.data.DataLoader(tr_eval_dataset, batch_size=val_batch_size,
                                                 shuffle=True, num_workers=0, pin_memory=True)
    te_eval_loader = torch.utils.data.DataLoader(te_eval_dataset, batch_size=val_batch_size,
                                                 shuffle=False, num_workers=0, pin_memory=True)

    # =========================
    # 1) SimCLR 사전학습
    # =========================
    simclr = SimCLRWrapper(base_model='resnet50', proj_out_dim=128).to(device)
    # optimizer_ssl = torch.optim.AdamW(simclr.parameters(), lr=learning_rate_ssl, weight_decay=1e-4)
    # scheduler_ssl = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ssl, T_max=num_epochs_ssl)
    optimizer_ssl = torch.optim.Adam(simclr.parameters(), weight_decay=5e-4, lr=0.001 )
    scheduler_ssl = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ssl, T_max=num_epochs_ssl)
    writer_ssl = SummaryWriter(log_dir=f'logs/{str_save}/ssl')

    simclr.train()
    for epoch in range(num_epochs_ssl):
        losses = []
        for (x1, x2, _) in tqdm(tr_ssl_loader, desc=f'[SSL] Epoch {epoch+1}/{num_epochs_ssl}'):
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)

            z1 = simclr(x1)
            z2 = simclr(x2)
            loss = simclr_nt_xent(z1, z2, temperature=temperature)

            optimizer_ssl.zero_grad()
            loss.backward()
            optimizer_ssl.step()
            losses.append(loss.item())

        scheduler_ssl.step()
        avg_loss = float(np.mean(losses))
        print(f'[SSL] epoch {epoch:03d} | loss {avg_loss:.4f} | lr {optimizer_ssl.param_groups[0]["lr"]:.6f}')
        writer_ssl.add_scalar('loss', avg_loss, epoch)

        # 주기적으로 백본 가중치 저장 (projection head 제외하고 저장할 수도 있음)
        if (epoch+1) % 50 == 0 or epoch+1 == num_epochs_ssl:
            torch.save(simclr.state_dict(), os.path.join(path_save, f'simclr_resnet50_epoch{epoch+1}.pth'))

    writer_ssl.flush()
    # 사전학습 완료 후, 백본 가중치만 추출
    # simclr.encoder → resnet50 backbone, simclr.proj → projection head
    # 체크포인트로부터 선형평가에 사용할 백본 파라미터를 복사
    backbone = FrozenBackbone('resnet50').to(device)
    # 가중치 복사
    with torch.no_grad():
        # simclr.encoder는 Sequential(children[:-1]) 구조와 동일
        # state_dict key 매칭을 위해 직접 복사
        bb_sd = backbone.encoder.state_dict()
        enc_sd = simclr.encoder.state_dict()
        for k in bb_sd.keys():
            if k in enc_sd:
                bb_sd[k] = enc_sd[k].clone()
        backbone.encoder.load_state_dict(bb_sd)

    # =========================
    # 2) 선형평가(백본 동결, Linear만 학습)
    # =========================
    classifier = LinearClassifier(in_dim=backbone.feat_dim, num_class=num_class).to(device)
    criterion_ce = nn.CrossEntropyLoss()
    # optimizer_lin = torch.optim.AdamW(classifier.parameters(), lr=learning_rate_lin, weight_decay=5e-4)
    # scheduler_lin = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_lin, T_max=num_epochs_lin)
    optimizer_lin = torch.optim.Adam(classifier.parameters(), weight_decay=5e-4, lr=0.001 )
    scheduler_lin = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_lin, T_max=num_epochs_lin)

    writer_tr = SummaryWriter(log_dir=f'logs/{str_save}/linear_train')
    writer_te = SummaryWriter(log_dir=f'logs/{str_save}/linear_val')
    writer_acc = SummaryWriter(log_dir=f'logs/{str_save}/linear_acc')

    best_acc = 0.0
    for epoch in range(num_epochs_lin):
        # --- train linear ---
        classifier.train()
        backbone.eval()  # 동결
        train_losses = []
        for x, y in tqdm(tr_eval_loader, desc=f'[Linear] Epoch {epoch+1}/{num_epochs_lin}'):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

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

        # --- eval ---
        classifier.eval()
        backbone.eval()
        val_losses, correct, total = [], 0, 0
        with torch.no_grad():
            for x, y in te_eval_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                feat = backbone(x)
                logits = classifier(feat)
                loss = criterion_ce(logits, y)
                val_losses.append(loss.item())

                pred = torch.argmax(logits, dim=1)
                total += y.size(0)
                correct += (pred == y).sum().item()

        acc = correct / total
        val_loss = float(np.mean(val_losses))
        print(f'[Linear] epoch {epoch:03d} | tr_loss {tr_loss:.4f} | val_loss {val_loss:.4f} | acc {acc:.4f}')
        writer_te.add_scalar('loss', val_loss, epoch)
        writer_acc.add_scalar('acc', acc, epoch)

        if acc > best_acc:
            best_acc = acc
            ckpt_path = os.path.join(path_save, f'linear_best_acc_{best_acc:.4f}.pth')
            torch.save({
                'backbone_state': backbone.state_dict(),
                'classifier_state': classifier.state_dict(),
                'acc': best_acc,
                'epoch': epoch
            }, ckpt_path)

    writer_tr.flush(); writer_te.flush(); writer_acc.flush()
    print('Done. Best linear eval Acc:', best_acc)
