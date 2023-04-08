import os


from PIL import Image
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from torchvision.models import densenet121
import torchvision.transforms as T

did = 0
device = torch.device(f"cuda:{did}")


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


def load_checkpoint(path):
    ckpt = torch.load(path, map_location=device)
    return ckpt


@torch.no_grad()
def evaluate(model, dataloader):
    model.eval()
    y_true = []
    y_pred = []
    for images, labels in tqdm(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        predictions = torch.argmax(probs, 1)
        y_true.extend(labels.cpu().numpy().tolist())
        y_pred.extend(predictions.cpu().numpy().tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted')
    print(
        f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}')
    cm = confusion_matrix(y_true, y_pred, labels=list(range(250)))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=range(250))
    _, axes = plt.subplots(figsize=(20, 15))
    disp.plot(ax=axes)
    plt.show()


if __name__ == "__main__":

    global_mean = [0.02362, 0.02362, 0.02362]
    global_std = [0.1114, 0.1114, 0.1114]

    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.Grayscale(3),
        T.ToTensor(),
        T.Normalize(global_mean, global_std)
    ])
    dataset = SketchDataset(df_dir="Data/Validation.csv",
                            imgs_dir="Data/Validation",
                            transform=transform)

    batch_size = 128
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=48)

    model = densenet121(num_classes=250).to(device)
    ckpt = load_checkpoint("best.pt")
    model.load_state_dict(ckpt['model'])

    evaluate(model, dataloader)
    plt.savefig("evaluation.png")
