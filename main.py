import torch
import torchvision.models as models
import pytorch_lightning as pl
from torch import nn, optim
import numpy as np
from PIL import Image
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
            self.df = df[df['image_id'].isin(train)].reset_index()
        elif split == 'val':
            self.df = df[df['image_id'].isin(val)].reset_index()
        elif split == 'test':
            self.df = df[df['image_id'].isin(test)].reset_index()
        else:
            raise ValueError('Nope')
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, 'image_id']
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path)

        attrs = ["Bags_Under_Eyes", "Bangs", "Black_Hair", "Blond_Hair", "Brown_Hair", "High_Cheekbones", "Mouth_Slightly_Open", "Chubby", "Eyeglasses", "Gray_Hair", "Narrow_Eyes", "Smiling", "Wearing_Hat"]

        # CelebA dataset labels are 1 for positive, -1 for negative.
        # Convert them to 1 for positive, 0 for negative.
        labels = torch.tensor(((self.df.loc[idx, attrs].values + 1) // 2).astype(np.float32))

        if self.transform:
            image = self.transform(image)

        return image, labels

# Define the transformations
transformations = Compose([
    CenterCrop(178),  # Center crop to remove the black edges (CelebA specific)
    Resize((128, 128)),  # Resize to 128x128
    RandAugment(3, 15),  # Apply RandAugment with N=3 and M=15
    ToTensor(),  # Convert PIL image to PyTorch tensor
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize to match the input range expected by ResNet
])

test_transform = Compose([
    CenterCrop(178),  # Center crop to remove the black edges (CelebA specific)
    Resize((128, 128)),  # Resize to 128x128
    ToTensor(),  # Convert PIL image to PyTorch tensor
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize to match the input range expected by ResNet
])

# Initialize the dataset
train_ds = CelebADataset("archive\img_align_celeba\img_align_celeba", split='train', transform=transformations)
val_ds = CelebADataset("archive\img_align_celeba\img_align_celeba", split='val', transform=test_transform)
test_ds = CelebADataset("archive\img_align_celeba\img_align_celeba", split='test', transform=test_transform)


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
        loss = sum(nn.BCELoss()(pred.flatten(), label) for pred, label in zip(preds, labels.T))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        preds = self.forward(images)
        loss = sum(nn.BCELoss()(pred.flatten(), label) for pred, label in zip(preds, labels.T))
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=4*1e-4)
        return optimizer


from torch.utils.data import DataLoader
from tqdm import tqdm

def test():
    test_dataloader = DataLoader(test_ds, batch_size=256, num_workers=3, pin_memory=True, persistent_workers=True)
    model = CelebAAttributeModel.load_from_checkpoint(r"F:\celeba\lightning_logs\version_13\checkpoints\epoch=61-step=39431.ckpt").cuda()

    all_labels = []
    all_preds = []
    for image, labels in tqdm(test_dataloader):
        with torch.no_grad():
            out = model(image.cuda())
            print(out.shape)


def main():
    # Create DataLoaders for the training and validation sets
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=3, persistent_workers=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=256, num_workers=3, persistent_workers=True, pin_memory=True)

    # Initialize the model
    model = CelebAAttributeModel()

    # Define any PyTorch Lightning callbacks you want to use
    callbacks = [
        pl.callbacks.ModelCheckpoint(save_top_k=1),  # Save the best model
        # Add more callbacks here if needed
    ]

    # Initialize the Trainer
    trainer = pl.Trainer(max_epochs=100, gpus=1, callbacks=callbacks, benchmark=True, )  

    # Train the model
    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    # main()
    test()