import argparse
import os

import torch
from torch.utils.data import Dataset, DataLoader

import pandas as pd
from tqdm import tqdm
from PIL import Image

from torchvision.models import densenet121
import torchvision.transforms as T


class SketchDataset(Dataset):
    def __init__(self, imgs_dir, transform=None):

        self.imgs_dir = imgs_dir
        self.imgs = os.listdir(imgs_dir)
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.imgs_dir))

    def __getitem__(self, idx):
        image_name = self.imgs_dir[idx]
        image_path = os.path.join(self.imgs_dir, image_name)
        img = Image.open(image_path).convert("L")

        # invert pixels
        img = T.functional.invert(img)

        if self.transform:
            img = self.transform(img)

        return img, image_name


def main(args):
    # mean and std for normalization
    train_mean = [0.02362, 0.02362, 0.02362]
    train_std = [0.1114, 0.1114, 0.1114]
    # configure device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # transformations
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.Grayscale(3),
        T.ToTensor(),
        T.Normalize(train_mean, train_std)
    ])
    # configure dataset and dataloader
    dataset = SketchDataset(args.src, transform=transform)
    dataloader = DataLoader(dataset)
    # instantiate model and load its weights
    model = densenet121(num_classes=250).to(device).eval()
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt['model'])
    # run inference
    df = pd.DataFrame(columns=["image_name", "category"])
    for image, name in tqdm(dataloader):
        image = image.to(device)
        # unpack
        name = name[0]
        # remove .png
        name = name[:-4]
        with torch.no_grad():
            logits = model(image)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item() + 1
        pred = int(pred)
        row = pd.DataFrame({"image_name": [name], "category": [pred]})
        df = pd.concat([df, row])
    df.to_csv(args.dst, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=str,
                        help="path to input data (source files (folder containing images))")
    parser.add_argument("dst", type=str,
                        help="destination (similar CSV file format as given in ground truth))")
    parser.add_argument("ckpt", type=str,
                        help="path to model checkpoint")
    args = parser.parse_args()

    main(args)
