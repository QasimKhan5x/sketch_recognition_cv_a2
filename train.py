import os
import random

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim import SGD

from torchvision.models import densenet121
import torchvision.transforms as T

dev_id = 0
torch.cuda.set_device(dev_id)
device = torch.device(f"cuda:{dev_id}")


def set_seed(seed):
    """Set seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class SketchDataset(Dataset):
    def __init__(self, df_dir, imgs_dir, transform=None):

        self.df = pd.read_csv(df_dir, names=["image_name", "category"])
        self.df['image_name'] = self.df['image_name'].map(str) + ".png"

        self.imgs_dir = imgs_dir
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        img_name = os.path.join(self.imgs_dir, self.df.loc[idx, "image_name"])
        img = Image.open(img_name).convert("L")
        # 1 - 250 should be 0 - 249
        label = self.df.loc[idx, "category"] - 1

        # invert pixels
        img = T.functional.invert(img)

        if self.transform:
            img = self.transform(img)

        return img, label


@torch.no_grad()
def validate(model, val_loader, criterion, epoch, writer=None):
    model.eval()
    val_loss = 0.0
    val_acc = 0.0

    for inputs, labels in tqdm(val_loader, total=len(val_loader)):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # B x 1 x 224 x 224 -> B x 250
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        val_loss += loss.item() * inputs.size(0)

        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        val_acc += torch.sum(preds == labels).item()

    val_loss /= len(val_loader.dataset)
    val_acc /= len(val_loader.dataset)

    if writer is not None:
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('Accuracy/validation', val_acc, epoch)

    return val_loss, val_acc


def train(model, train_loader, criterion, optimizer, scheduler, writer, epoch):
    model.train()
    train_loss = 0.0
    train_acc = 0.0

    for inputs, labels in tqdm(train_loader, total=len(train_loader)):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # five crop support
        bs, ncrops, c, h, w = inputs.size()
        # fuse batch size and ncrops
        outputs = model(inputs.view(-1, c, h, w))
        # avg over crops
        outputs_avg = outputs.view(bs, ncrops, -1).mean(1)

        # outputs = model(inputs)
        loss = criterion(outputs_avg, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss += loss.item() * inputs.size(0)
        probs = torch.softmax(outputs_avg, dim=1)
        preds = torch.argmax(probs, dim=1)
        train_acc += torch.sum(preds == labels).item()

    train_loss /= len(train_loader.dataset)
    train_acc /= len(train_loader.dataset)

    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Accuracy/train', train_acc, epoch)

    print(f'Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}')

    return train_loss, train_acc


def save_checkpoint(path, model, optimizer, scheduler, epoch):
    if not os.path.isdir(path.split('/')[0]):
        os.mkdir(path.split('/')[0])
    torch.save({
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch
    }, path)


def load_checkpoint(path):
    ckpt = torch.load(path, map_location=device)
    return ckpt


def plot_loss_acc(train_loss, train_acc, val_loss, val_acc, path):
    """
    Plot the training and validation loss and accuracy curves and save the figure to a file.

    Args:
        train_loss (list): List of training loss values.
        train_acc (list): List of training accuracy values.
        val_loss (list): List of validation loss values.
        val_acc (list): List of validation accuracy values.
        path (str): Path to save the figure.
    """
    # Create a subplot with 1 row and 2 columns
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    # Plot the training and validation losses
    axs[0].plot(train_loss, label='train')
    axs[0].plot(val_loss, label='val')
    axs[0].set_title('Training and Validation Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    # Plot the training and validation accuracies
    axs[1].plot(train_acc, label='train')
    axs[1].plot(val_acc, label='val')
    axs[1].set_title('Training and Validation Accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()

    # Save the figure to the specified path
    fp = os.path.join(path, "plot.png")
    if os.path.isfile(fp):
        plt.savefig(os.path.join(path, "plot_ft.png"))
    else:
        plt.savefig(fp)


def train_and_validate(exp_name, model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, val_step):
    writer = SummaryWriter(log_dir=f"runs/{exp_name}")
    best_val_acc = 0.0
    best_epoch = 0
    train_loss_history, train_acc_history, val_loss_history, val_acc_history = [], [], [], []
    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        print(f'Epoch {epoch+1}\n-------------------------------')
        train_loss, train_acc = train(model, train_loader, criterion,
                                      optimizer, scheduler, writer, epoch)
        val_loss, val_acc = validate(
            model, val_loader, criterion, epoch, writer)
        if val_acc > best_val_acc:
            save_checkpoint(f'{exp_name}/best.pt', model,
                            optimizer, scheduler, epoch)
            best_val_acc = val_acc
            best_epoch = epoch
        if epoch == 0 or (epoch+1) % val_step == 0:
            print(
                f'Validation Loss: {val_loss:.4f}  Validation Acc: {val_acc:.4f}, Best: {best_val_acc:.4f} ({best_epoch+1})')

        # save loss and acc
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)
        writer.flush()
        save_checkpoint(f'{exp_name}/last.pt', model,
                        optimizer, scheduler, epoch)

    writer.close()
    plot_loss_acc(train_loss_history, train_acc_history,
                  val_loss_history, val_acc_history, exp_name)
    return model


class FiveCropTransform:
    def __init__(self, size):
        self.crop_transform = T.FiveCrop(size)

    def __call__(self, img):
        crops = self.crop_transform(img)
        return [T.Compose([
            T.Grayscale(3),
            T.RandomHorizontalFlip(),
            T.RandomAffine(35, translate=(0, 1/7)),
            T.ToTensor(),
            T.Normalize(global_mean, global_std),
            T.RandomErasing(p=0.2)
        ])(crop) for crop in crops]


if __name__ == "__main__":
    set_seed(42)

    global_mean = [0.02362, 0.02362, 0.02362]
    global_std = [0.1114, 0.1114, 0.1114]

    train_transform = T.Compose([
        T.Resize(256),
        FiveCropTransform(224),
        T.Lambda(lambda crops: torch.stack(crops))
    ])
    valid_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.Grayscale(3),
        T.ToTensor(),
        T.Normalize(global_mean, global_std)
    ])
    train_ds = SketchDataset(df_dir="Data/Train.csv",
                             imgs_dir="Data/Train",
                             transform=train_transform)
    valid_ds = SketchDataset(df_dir="Data/Validation.csv",
                             imgs_dir="Data/Validation",
                             transform=valid_transform)
    print("Number of training examples:", len(train_ds))
    print("Number of validation examples:", len(valid_ds))
    batch_size = 128
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          pin_memory=True, num_workers=24)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, num_workers=24)

    output_shape = 250
    model = densenet121(pretrained=True)
    for param in model.features.parameters():
        param.requires_grad = False
    model.classifier = nn.Linear(1024, output_shape)
    model = model.to(device)

    num_epochs = 60
    # Define loss, optimizer, and lr scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=3e-3,
                    momentum=0.9, weight_decay=5e-4)
    scheduler = OneCycleLR(optimizer, max_lr=0.1,
                           epochs=num_epochs, steps_per_epoch=len(train_dl))

    exp1 = 'densenet_fc+fivecrop'
    # model = train_and_validate(exp1, model, train_dl, valid_dl, criterion,
    #                            optimizer, scheduler, num_epochs, val_step=1)

    # ============= experiment 2 ====================
    exp2 = 'densenet_fc->full+fivecrop'
    batch_size = 32
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          pin_memory=True, num_workers=24)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, num_workers=24)
    ckpt = load_checkpoint(exp1 + "/best.pt")
    model.load_state_dict(ckpt['model'])
    for param in model.parameters():
        param.requires_grad = True
    optimizer = SGD(model.parameters(), lr=3e-4,
                    momentum=0.9, weight_decay=5e-5)
    scheduler = OneCycleLR(optimizer, max_lr=0.01,
                           epochs=num_epochs, steps_per_epoch=len(train_dl))
    model = train_and_validate(exp2, model, train_dl, valid_dl,
                               criterion, optimizer, scheduler, num_epochs,
                               val_step=1)
