import albumentations
import cv2
import torch
from torch.autograd import Variable

from dataset.datasets_base import DatasetAcne04Class
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import random
import torch.nn.functional as F
from torchvision import models as models_tv

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

seed = 3407
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device :', device)
    print('Current cuda device:', torch.cuda.current_device())
    print('Count of using GPUs:', torch.cuda.device_count())

    num_epochs = 50
    train_batch_size = 16
    val_batch_size = 8
    learning_rate = 0.001
    num_class = 200

    len_h = 224
    len_w = 224
    color = cv2.IMREAD_COLOR_RGB
    colorspace = None

    str_save = 'Acne04_RGB_Self_0.40'

    path_tr_src = '/shared/data/ACNE04_Total/Semi-supervised/0.40_Tr_labeled'
    path_te_src = '/shared/data/ACNE04_Total/Cropped Faces/Te'

    path_save = 'save_model/%s' % str_save
    if os.path.isdir(path_save) == False:
        os.makedirs(path_save)

    transform_tr = albumentations.Compose([
        albumentations.RandomBrightnessContrast(p=0.5),  # 밝기/대비 변화
        # albumentations.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),  # 색조/채도/명도 변화
        albumentations.HorizontalFlip(p=0.5),
        albumentations.Rotate(limit=45, p=0.5),
        albumentations.Resize(height=len_h, width=len_w)
    ])

    transforms_te = albumentations.Compose([
        albumentations.Resize(height=len_h, width=len_w)
    ])

    tr_dataset = DatasetAcne04Class(path_src=path_tr_src, transforms=transform_tr, color=color, colorspace=colorspace)
    # tr_loader = torch.utils.data.DataLoader(dataset=tr_dataset, batch_size=train_batch_size, shuffle=True, num_workers=4,
    #                                         pin_memory=True, persistent_workers=True)
    tr_loader = torch.utils.data.DataLoader(dataset=tr_dataset, batch_size=train_batch_size, shuffle=True,
                                            num_workers=4,
                                            pin_memory=True, persistent_workers=False)

    te_dataset = DatasetAcne04Class(path_src=path_te_src, transforms=transforms_te, color=color, colorspace=colorspace)
    # te_loader = torch.utils.data.DataLoader(dataset=te_dataset, batch_size=train_batch_size, shuffle=False, num_workers=4,
    #                                         pin_memory=True, persistent_workers=True)
    te_loader = torch.utils.data.DataLoader(dataset=te_dataset, batch_size=train_batch_size, shuffle=False,
                                            num_workers=4,
                                            pin_memory=True, persistent_workers=False)

    # load model
    model = models_tv.resnet50(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, num_class)

    # 🔽 SimCLR 인코더 가중치 로드
    ckpt_path = '/shared/data/ACNE04_Total/Acne04_best/SimCLR_Acne04_0.40/simclr_resnet50_best.pth'

    state = torch.load(ckpt_path, map_location="cpu")

    # state는 딕셔너리이므로, 실제 인코더 가중치가 담긴 'encoder_state_dict'를 추출해야 합니다.
    encoder_weights = state['encoder_state_dict']

    # fc는 분류용으로 새로 만들었기 때문에 strict=False로 로드합니다.
    missing, unexpected = model.load_state_dict(encoder_weights, strict=False)  # <--- 수정된 부분
    print("Missing keys:", missing)  # ['fc.weight', 'fc.bias'] 만 출력되어야 정상입니다.
    print("Unexpected keys:", unexpected)  # [] (빈 리스트)가 출력되어야 정상입니다.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), weight_decay=5e-4, lr=0.001 )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    writer_loss_tr = SummaryWriter(log_dir='logs/%s/loss_tr' % str_save)
    writer_loss_te = SummaryWriter(log_dir='logs/%s/loss_te' % str_save)

    writer_loss_acc = SummaryWriter(log_dir='logs/%s/accuracy' % str_save)

    epsilon = 2.22045e-16
    max_acc = 0

    criterion = torch.nn.CrossEntropyLoss()


    for epoch in range(num_epochs):
        model.train()
        batch_losses = []

        for step, (inp_src, inp_lbl) in enumerate(tqdm(tr_loader)):

            inp_src = Variable(inp_src.type(torch.FloatTensor)).to(device)
            inp_lbl = inp_lbl.to(device)
            outputs = model(inp_src)

            loss = criterion(outputs, inp_lbl)
            loss_value = loss.data.cpu().numpy()
            batch_losses.append(loss_value)

            optimizer.zero_grad()  # (reset gradients)
            loss.backward()  # (compute gradients)
            optimizer.step()  # (perform optimization step)

        epoch_loss = np.mean(batch_losses)
        print('epoch %d: train loss = %.8f, lr = %f' % (epoch, epoch_loss, optimizer.param_groups[0]['lr']))
        writer_loss_tr.add_scalar('loss', epoch_loss, epoch)
        scheduler.step()

        model.eval()  # (set in evaluation mode, this affects BatchNorm and dropout)

        batch_losses = []
        correct = 0
        total = 0
        with torch.no_grad():
            for step, (inp_src, inp_lbl) in enumerate(tqdm(te_loader)):

                inp_src = Variable(inp_src.type(torch.FloatTensor)).to(device)
                inp_lbl = inp_lbl.to(device)

                outputs = model(inp_src)

                loss = criterion(outputs, inp_lbl)
                loss_value = loss.data.cpu().numpy()
                batch_losses.append(loss_value)

                # 정확도 계산
                _, predicted = torch.max(outputs.data, 1)  # 가장 확률 높은 클래스
                total += inp_lbl.size(0)
                correct += (predicted == inp_lbl).sum().item()

        accuracy = correct / total
        epoch_loss = np.mean(batch_losses)
        print('epoch %d: test loss = %.8f, accuracy = %.4f' % (epoch, epoch_loss, accuracy))

        writer_loss_te.add_scalar('loss', epoch_loss, epoch)
        writer_loss_acc.add_scalar('acc', accuracy, epoch)

        if (max_acc < accuracy):
            max_acc = accuracy
            fns_check = '%s/resnet50_epoch_%d_acc_%.4f.pth' % (path_save, epoch, max_acc)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss_value,
            }, fns_check)

        writer_loss_tr.flush()
        writer_loss_te.flush()
        writer_loss_acc.flush()