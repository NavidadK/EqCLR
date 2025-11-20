# eqCLR_pmnist
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms

import numpy as np
import time
import pickle

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from medmnist import PathMNIST

from eqCLR.eq_resnet import EqResNet18

###################### PARAMS ##############################

BACKBONE = "resnet18"

BATCH_SIZE = 512
N_EPOCHS = 100 # 1000
N_CPU_WORKERS = 16
BASE_LR = 0.03         # important
WEIGHT_DECAY = 5e-4    # important
MOMENTUM = 0.9
PROJECTOR_HIDDEN_SIZE = 1024
PROJECTOR_OUTPUT_SIZE = 128
CROP_LOW_SCALE = 0.2
GRAYSCALE_PROB = 0.1   # important
PRINT_EVERY_EPOCHS = 1

MODEL_FILENAME = f"{np.random.randint(10000):04}-path_mnist-eqCLR_resnet18_with_rotation"

###################### DATA LOADER #########################

pmnist_train = PathMNIST(split='train', download=False, size=28, root='data/pathmnist/', transform=transforms.ToTensor())
pmnist_test = PathMNIST(split='test', download=False, size=28, root='data/pathmnist/', transform=transforms.ToTensor())

print("Data loaded.")

# additional rotation
class RandomRightAngleRotation:
    """Randomly rotate PIL image by 90, 180, or 270 degrees."""
    def __call__(self, x):
        angle = int(torch.randint(1, 4, ()).item()) * 90
        return x.rotate(angle)

transforms_ssl = transforms.Compose(
    [
        transforms.RandomResizedCrop(size=32, scale=(CROP_LOW_SCALE, 1)),
        RandomRightAngleRotation(), # additional rotation
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply(
            [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
        ),
        transforms.RandomGrayscale(p=GRAYSCALE_PROB),
        transforms.ToTensor(), # NB: runtime faster when this line is last
    ]
)

class PairedTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return (self.transform(x), self.transform(x))


paired_ssl_transforms = PairedTransform(transforms_ssl)

pmnist_train_ssl = PathMNIST(split='train', download=False, size=28, root='data/pathmnist/', transform=paired_ssl_transforms)

pmnist_loader_ssl = DataLoader(
    pmnist_train_ssl,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=N_CPU_WORKERS,
    pin_memory=True,
)

###################### NETWORK ARCHITECTURE #########################

def infoNCE(features, temperature=0.5):
    x = F.normalize(features)
    cos_xx = x @ x.T / temperature
    cos_xx.fill_diagonal_(float("-inf"))
    
    batch_size = cos_xx.size(0) // 2
    targets = torch.arange(batch_size * 2, dtype=int, device=cos_xx.device)
    targets[:batch_size] += batch_size
    targets[batch_size:] -= batch_size

    return F.cross_entropy(cos_xx, targets)

model = EqResNet18(N=4)

optimizer = SGD(
    model.parameters(),
    lr=BASE_LR * BATCH_SIZE / 256,
    momentum=MOMENTUM,
    weight_decay=WEIGHT_DECAY,
)

scheduler = CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

# # Adam works almost as well but requires 1e-6 weight decay and warmup
# optimizer = Adam(model.parameters(), lr=3e-3, weight_decay=1e-6)
# 
# scheduler = SequentialLR(
#     optimizer,
#     schedulers=[
#         LinearLR(optimizer, start_factor=0.1, total_iters=10),
#         CosineAnnealingLR(optimizer, T_max=N_EPOCHS - 10),
#     ],
#     milestones=[10],
# )

###################### TRAINING LOOP #########################

print("Starting training.")

device = "cuda"

model.to(device)
model.train()
training_start_time = time.time()

for epoch in range(N_EPOCHS):
    epoch_loss = 0.0
    start_time = time.time()

    for batch_idx, batch in enumerate(pmnist_loader_ssl):
        views, _ = batch

        views = [view.to(device, non_blocking=True) for view in views]

        optimizer.zero_grad()

        _, z1 = model(views[0])
        _, z2 = model(views[1])
        loss = infoNCE(torch.cat((z1, z2)))
        epoch_loss += loss.item()

        loss.backward()
        optimizer.step()

    end_time = time.time()
    if (epoch + 1) % PRINT_EVERY_EPOCHS == 0:
        print(
            f"Epoch {epoch + 1}, "
            f"average loss {epoch_loss / len(pmnist_loader_ssl):.4f}, "
            f"{end_time - start_time:.1f} s",
            flush=True
        )

    scheduler.step()

training_end_time = time.time()
hours = (training_end_time - training_start_time) / 60 // 60
minutes = (training_end_time - training_start_time) / 60 % 60
average = (training_end_time - training_start_time) / N_EPOCHS
print(
    f"Total training length for {N_EPOCHS} epochs: {hours:.0f}h {minutes:.0f}min",
    f"({average:.1f} sec/epoch)",
    flush=True
)

model.eval()
torch.save(model.state_dict(), f'results/model_weights/{MODEL_FILENAME}.pt')
print(f"Model saved to {MODEL_FILENAME}")

model_details = {
    "Filename": MODEL_FILENAME,
    "N_EPOCHS": N_EPOCHS,
    "BATCH_SIZE": BATCH_SIZE,
    "BASE_LR": BASE_LR,
    "WEIGHT_DECAY": WEIGHT_DECAY,
    "MOMENTUM": MOMENTUM,
    "CROP_LOW_SCALE": CROP_LOW_SCALE,
    "GRAYSCALE_PROB": GRAYSCALE_PROB,
    "model_state_dict": model.state_dict(),
    "BACKBONE": BACKBONE,
    "PROJECTOR_HIDDEN_SIZE": PROJECTOR_HIDDEN_SIZE,
    "PROJECTOR_OUTPUT_SIZE": PROJECTOR_OUTPUT_SIZE,
    "Time": training_end_time - training_start_time,
}

with open(f'results/model_details/{MODEL_FILENAME}_details.pkl', 'wb') as f:
    pickle.dump(model_details, f)
    
print(f"Model details saved to {MODEL_FILENAME}_details.pkl")
