import torch
import torchvision.models as models
import pytorch_lightning as pl
from torch import nn, optim

import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import Compose, Resize, Normalize, ToTensor, RandomCrop, CenterCrop
from torchvision.transforms import RandAugment
import os 

class CelebADataset(Dataset):
    def __init__(self, img_dir, split='train', transform=None):
        self.img_dir = img_dir
        df_part = pd.read_csv(r"F:\celeba\archive\list_eval_partition.csv")

        train = df_part[df_part['partition'] == 0]['image_id'].tolist()
        val = df_part[df_part['partition'] == 1]['image_id'].tolist()
        test = df_part[df_part['partition'] == 2]['image_id'].tolist()
        df = pd.read_csv(r"F:\celeba\archive\list_attr_celeba.csv")

        if split == 'train':
            self.df = df[df['image_id'].isin(train)]
        elif split == 'val':
            self.df = df[df['image_id'].isin(val)]
        elif split == 'test':
            self.df = df[df['image_id'].isin(test)]
        else:
            raise ValueError('Nope')
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, img_name)
        image = read_image(img_path)

        # CelebA dataset labels are 1 for positive, -1 for negative.
        # Convert them to 1 for positive, 0 for negative.
        labels = torch.tensor((self.df.iloc[idx, 1:].values + 1) // 2)

        if self.transform:
            image = self.transform(image)

        return image, labels

# Define the transformations
transformations = Compose([
    CenterCrop(178),  # Center crop to remove the black edges (CelebA specific)
    Resize((128, 128)),  # Resize to 128x128
    ToTensor(),  # Convert PIL image to PyTorch tensor
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize to match the input range expected by ResNet
    RandAugment(3, 15)  # Apply RandAugment with N=3 and M=15
])

# Initialize the dataset
train_ds = CelebADataset("archive\img_align_celeba\img_align_celeba", split='train', transform=transformations)
val_ds = CelebADataset("archive\img_align_celeba\img_align_celeba", split='val', transform=transformations)
test_ds = CelebADataset("archive\img_align_celeba\img_align_celeba", split='test', transform=transformations)


class CelebAAttributeModel(pl.LightningModule):
    def __init__(self):
        super(CelebAAttributeModel, self).__init__()
        # Load pre-trained ResNet50
        resnet = models.resnet50(pretrained=True)
        num_ftrs = resnet.fc.in_features
        resnet.fc = nn.Identity() # Remove the final layer
        
        self.features = resnet
        self.classifiers = nn.ModuleList([nn.Linear(num_ftrs, 1) for _ in range(13)])

    def forward(self, x):
        x = self.features(x)
        return [torch.sigmoid(c(x)) for c in self.classifiers]

    def training_step(self, batch, batch_idx):
        images, labels = batch
        preds = self.forward(images)
        loss = sum(nn.BCELoss()(pred, label) for pred, label in zip(preds, labels.T))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        preds = self.forward(images)
        loss = sum(nn.BCELoss()(pred, label) for pred, label in zip(preds, labels.T))
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

# Initialize the dataset
celeba_dataset = CelebADataset("path_to_your_data/img_align_celeba", "path_to_your_data/list_attr_celeba.csv", transform=transformations)

# Split the dataset into train and validation sets
train_size = int(0.8 * len(celeba_dataset))
val_size = len(celeba_dataset) - train_size
train_dataset, val_dataset = random_split(celeba_dataset, [train_size, val_size])

# Create DataLoaders for the training and validation sets
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Initialize the model
model = CelebAAttributeModel()

# Define any PyTorch Lightning callbacks you want to use
callbacks = [
    # Example: pl.callbacks.ModelCheckpoint(save_top_k=1),  # Save the best model
    # Add more callbacks here if needed
]

# Initialize the Trainer
trainer = pl.Trainer(max_epochs=10, gpus=1, callbacks=callbacks)  # Train for 10 epochs, adjust as needed

# Train the model
trainer.fit(model, train_loader, val_loader)