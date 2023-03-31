import os
import random
from pprint import pprint

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import GradientBoostingClassifier

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as T

dev_id = 2
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

        self.df = pd.read_csv(df_dir)
        
        if "Train" in df_dir:
            self.df = self.df.rename(columns={"1": "image_name", "1.1": "category"})
        elif "Validation" in df_dir:
            self.df = self.df.rename(columns={"41": "image_name", "1": "category"})

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

if __name__ == "__main__":
    set_seed(42)
    
    global_mean = [0.02362, 0.02362, 0.02362]
    global_std = [0.1114, 0.1114, 0.1114]
    
    train_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.Grayscale(3),
        T.RandomHorizontalFlip(),
        T.RandomAffine(35, translate=(0, 1/7)),
        T.ToTensor(),
        T.Normalize(global_mean, global_std),
        T.RandomErasing(p=0.2)
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
    batch_size = 128
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                          pin_memory=True, num_workers=24)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, num_workers=24)
    
    output_shape = 250
    model = torchvision.models.densenet121(num_classes=output_shape)
    model = model.to(device)
    ckpt = load_checkpoint("densenet_fc->full/best.pt")['model']
    model.load_state_dict(ckpt)
    
    # Remove the last fully connected layer
    densenet121 = nn.Sequential(*list(model.children())[:-1])
    # Set the model to evaluation mode
    densenet121.eval()

    # Define the feature extractor function
    @torch.no_grad()
    def extract_features(x):
        features = densenet121(x.cuda(dev_id))
        batch_size = features.size(0)
        feature_size = features.size(1)
        features = features.view(batch_size, feature_size, -1).mean(-1)
        return features.cpu().numpy()
    
    x_train = np.zeros((len(train_ds), 1024))
    y_train = np.zeros(len(train_ds))

    x_valid = np.zeros((len(valid_ds), 1024))
    y_valid = np.zeros(len(valid_ds))

    print("Extracting train features")
    for i, (images, targets) in enumerate(tqdm(train_dl)):
        x_train[i * 128:(i + 1) * 128] = extract_features(images)
        y_train[i * 128:(i + 1) * 128] = targets.numpy()

    print("Extracting validation features")
    for i, (images, targets) in enumerate(tqdm(valid_dl)):
        x_valid[i * 128:(i + 1) * 128] = extract_features(images)
        y_valid[i * 128:(i + 1) * 128] = targets.numpy()
        
    # free memory
    model = model.cpu()
    del model
    torch.cuda.empty_cache()
    
    print("Tuning SVC")
    # Perform grid search for SVM hyperparameters
    svm_param_grid = {'C': [0.01, 0.1, 1, 10, 100], 
                    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                    'degree': [2, 3, 4, 5],
                    'gamma': ['scale', 'auto'] + [0.01, 0.1, 1, 10],
                    'coef0': [-1, 0, 1]}
    svm = SVC(random_state=42)
    svm_grid_search = GridSearchCV(svm, svm_param_grid, cv=5, verbose=2)
    svm_grid_search.fit(x_train, y_train)
    svm_best_params = svm_grid_search.best_params_
    # Train SVM with best hyperparameters
    print("Training SVM with best hyperparameters")
    svm_best = SVC(random_state=42, **svm_best_params, verbose=True)
    svm_best.fit(x_train, y_train)
    # Compute the accuracy of the classifier
    valid_preds = svm_best.predict(x_valid)
    svm_acc = accuracy_score(y_valid, valid_preds)
    
    # Perform grid search for LinearSVC hyperparameters
    print("Tuning LinearSVC")
    linear_svc_param_grid = {'C': [0.01, 0.1, 1, 10, 100],
                            'loss': ['hinge', 'squared_hinge'],
                            'dual': [True, False],
                            'penalty': ['l1', 'l2'],
                            'tol': [1e-5, 1e-4, 1e-3]}
    linear_svc = LinearSVC(random_state=42, max_iter=10000)
    linear_svc_grid_search = GridSearchCV(linear_svc, linear_svc_param_grid, cv=5, verbose=2)
    linear_svc_grid_search.fit(x_train, y_train)
    linear_svc_best_params = linear_svc_grid_search.best_params_
    # Train LinearSVC with best hyperparameters
    linear_svc_best = LinearSVC(random_state=42, verbose=2, **linear_svc_best_params)
    linear_svc_best.fit(x_train, y_train)
    linear_svc_pred = linear_svc_best.predict(x_valid)
    linear_svc_acc = accuracy_score(y_valid, linear_svc_pred)
    
    print("Tuning GBM")
    # Perform grid search for GBM hyperparameters
    gbm_param_grid = {'learning_rate': [0.01, 0.1, 0.5, 1],
                    'n_estimators': [50, 100, 200, 500],
                    'max_depth': [2, 3, 4, 5, 6],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'subsample': [0.5, 0.8, 1.0],
                    'max_features': ['sqrt', 'log2', None]}
    gbm = GradientBoostingClassifier(random_state=42)
    gbm_grid_search = GridSearchCV(gbm, gbm_param_grid, cv=5, verbose=2)
    gbm_grid_search.fit(x_train, y_train)
    gbm_best_params = gbm_grid_search.best_params_
    # Train GBM with best hyperparameters
    gbm_best = GradientBoostingClassifier(random_state=42, verbose=2, **gbm_best_params)
    gbm_best.fit(x_train, y_train)
    gbm_pred = gbm_best.predict(x_valid)
    gbm_acc = accuracy_score(y_valid, gbm_pred)

    print(f"SVM accuracy: {svm_acc:.3f}")
    pprint(f"Best SVM hyperparameters: {svm_best_params}")
    print(f"LinearSVC accuracy: {linear_svc_acc:.3f}")
    pprint(f"Best LinearSVC hyperparameters: {linear_svc_best_params}")
    print(f"GBM accuracy: {gbm_acc:.3f}")
    pprint(f"Best GBM hyperparameters: {gbm_best_params}")

    
    
    