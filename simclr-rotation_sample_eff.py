import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import Dataset, DataLoader

from torchvision.datasets import CIFAR10
from torchvision.models import resnet18, resnet34, resnet50
import torchvision.transforms as transforms
from torch.utils.data import Subset

import numpy as np
import time

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from medmnist import PathMNIST
import pickle
from evaluation import model_eval, eval_knn_single, dataset_to_X_y

###################### PARAMS ##############################

BACKBONE = "resnet18"

BATCH_SIZE = 512
N_EPOCHS = 100 # 1000
N_CPU_WORKERS = 16
BASE_LR = 0.06         # important (0.03)
WEIGHT_DECAY = 5e-4    # important
MOMENTUM = 0.9
PROJECTOR_HIDDEN_SIZE = 1024
PROJECTOR_OUTPUT_SIZE = 128
CROP_LOW_SCALE = 0.2
GRAYSCALE_PROB = 0.1   # important
PRINT_EVERY_EPOCHS = 1
EVAL_DURING_TRAIN = False
ITER_SAVE_EMBED = None
IMG_RESIZE = 33  # if None, use original size 28x28
MAXPOOL = True

MODEL_FILENAME = f"2113-path_mnist-{BACKBONE}"

###################### DATA LOADER #########################

if IMG_RESIZE is not None:
    transform = transforms.Compose([
        transforms.Resize((IMG_RESIZE, IMG_RESIZE)),
        transforms.ToTensor(),
    ])
else:
    transform = transforms.ToTensor()

pmnist_train = PathMNIST(split='train', download=False, size=28, root='data/pathmnist/', transform=transform)
pmnist_test = PathMNIST(split='test', download=False, size=28, root='data/pathmnist/', transform=transform)
print("Data loaded.")

if IMG_RESIZE is None:
    IMG_RESIZE = pmnist_train[0][0].shape[1]  # get image size from dataset
print(f"Image size (resized): {IMG_RESIZE}")

# additional rotation
class RandomRightAngleRotation:
    """Randomly rotate PIL image by 90, 180, or 270 degrees."""
    def __call__(self, x):
        angle = int(torch.randint(1, 4, ()).item()) * 90
        return x.rotate(angle)

transforms_ssl = transforms.Compose(
    [
        transforms.Resize((IMG_RESIZE, IMG_RESIZE)),
        transforms.RandomResizedCrop(size=IMG_RESIZE, scale=(CROP_LOW_SCALE, 1)),
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
    
def make_subset(dataset, fraction, seed=0):
    assert 0 < fraction <= 1.0
    n_total = len(dataset)
    n_keep = int(fraction * n_total)

    rng = np.random.default_rng(seed)
    indices = rng.choice(n_total, size=n_keep, replace=False)

    return Subset(dataset, indices)

fractions = [0.6, 0.8]

for frac in fractions:
    print(f"\n--- Training with fraction {frac} of the data ---\n")

    paired_ssl_transforms = PairedTransform(transforms_ssl)

    pmnist_train_ssl = PathMNIST(split='train', download=False, size=28, root='data/pathmnist/', transform=paired_ssl_transforms)

    pmnist_train_ssl_frac = make_subset(pmnist_train_ssl, fraction=frac, seed=42)

    pmnist_loader_ssl = DataLoader(
        pmnist_train_ssl_frac,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=N_CPU_WORKERS,
        pin_memory=True,
    )

    ###################### NETWORK ARCHITECTURE #########################

    class ResNetwithProjector(nn.Module):
        def __init__(self, backbone_network, maxpool=MAXPOOL):
            super().__init__()

            self.backbone = backbone_network(weights=None)
            self.backbone_output_dim = self.backbone.fc.in_features

            if not maxpool:
                self.backbone.conv1 = nn.Conv2d(
                    3, 64, kernel_size=3, stride=1, padding=1, bias=False
                )
                self.backbone.maxpool = nn.Identity()
                
            self.backbone.fc = nn.Identity()

            self.projector = nn.Sequential(
                nn.Linear(self.backbone_output_dim, PROJECTOR_HIDDEN_SIZE), 
                nn.ReLU(), 
                nn.Linear(PROJECTOR_HIDDEN_SIZE, PROJECTOR_OUTPUT_SIZE),
            )

        def forward(self, x):
            h = self.backbone(x)
            z = self.projector(h)
            return h, z


    def infoNCE(features, temperature=0.5):
        x = F.normalize(features)
        cos_xx = x @ x.T / temperature
        cos_xx.fill_diagonal_(float("-inf"))
        
        batch_size = cos_xx.size(0) // 2
        targets = torch.arange(batch_size * 2, dtype=int, device=cos_xx.device)
        targets[:batch_size] += batch_size
        targets[batch_size:] -= batch_size

        return F.cross_entropy(cos_xx, targets)

    backbones = {
    "resnet18": resnet18,    # backbone_output_dim = 512
    "resnet34": resnet34,    # backbone_output_dim = 512
    "resnet50": resnet50,    # backbone_output_dim = 2048
    }

    model = ResNetwithProjector(backbones[BACKBONE])

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

    # ###################### TRAINING LOOP #########################

    # print("Starting training.")

    device = "cuda"
    model.to(device)

    model.train()
    knn_dict = {}
    embed_dict = {}
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

        scheduler.step()
        if EVAL_DURING_TRAIN:
            model.eval()
            with torch.no_grad():
                X_train, y_train, Z_train = dataset_to_X_y(pmnist_train, model)
                X_test, y_test, Z_test = dataset_to_X_y(pmnist_test, model)

                knn_acc = eval_knn_single(X_train, y_train, X_test, y_test)
                knn_dict[epoch] = knn_acc

                if ITER_SAVE_EMBED is not None and (epoch + 1) % ITER_SAVE_EMBED == 0:
                    embed_dict[epoch] = {
                        "X_test": X_test,
                        "y_test": y_test,
                    }
            model.train()

        if (epoch + 1) % PRINT_EVERY_EPOCHS == 0:
            print(
                f"Epoch {epoch + 1}, "
                f"average loss {epoch_loss / len(pmnist_loader_ssl):.4f}, "
                f"{end_time - start_time:.1f} s",
                f"KNN accuracy {knn_dict.get(epoch, 'N/A')}",
                flush=True
            )

    training_end_time = time.time()
    hours = (training_end_time - training_start_time) / 60 // 60
    minutes = (training_end_time - training_start_time) / 60 % 60
    average = (training_end_time - training_start_time) / N_EPOCHS
    print(
        f"Total training length for {N_EPOCHS} epochs: {hours:.0f}h {minutes:.0f}min",
        f"({average:.1f} sec/epoch)",
        flush=True
    )

    torch.save(model.state_dict(), f'results/model_weights/{MODEL_FILENAME}_frac{frac}_weights.pt')
    print(f"Model saved to {MODEL_FILENAME}_frac{frac}_weights.pt")

    model_details = {
        "Filename": MODEL_FILENAME,
        "Model": str(model),
        "N_EPOCHS": N_EPOCHS,
        "BATCH_SIZE": BATCH_SIZE,
        "BASE_LR": BASE_LR,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "MOMENTUM": MOMENTUM,
        "CROP_LOW_SCALE": CROP_LOW_SCALE,
        "GRAYSCALE_PROB": GRAYSCALE_PROB,
        "PROJECTOR_HIDDEN_SIZE": PROJECTOR_HIDDEN_SIZE,
        "PROJECTOR_OUTPUT_SIZE": PROJECTOR_OUTPUT_SIZE,
        "Training augmentations": transforms_ssl,
        "Training time": training_end_time - training_start_time,
        "Training time per epoch": average,
        "KNN during training": knn_dict,
        "Image resize": IMG_RESIZE,
        "Embeddings during training": embed_dict,
    }

    with open(f'results/model_details/{MODEL_FILENAME}_frac{frac}_details.pkl', 'wb') as f:
        pickle.dump(model_details, f)
        
    print(f"Model details saved to {MODEL_FILENAME}_frac{frac}_details.pkl")

    ###################### EVALUATION #########################

    # # # load weights
    # model.load_state_dict(torch.load(f'results/model_weights/2109-path_mnist-resnet18_frac0.4_weights.pt', weights_only=True))
    # print('Weights loaded.')

    model.eval()

    transforms_classifier = transforms.Compose(
        [
            transforms.Resize((IMG_RESIZE, IMG_RESIZE)),
            transforms.RandomResizedCrop(size=IMG_RESIZE, scale=(CROP_LOW_SCALE, 1)),
            transforms.RandomHorizontalFlip(),
            RandomRightAngleRotation(), # additional rotation
            transforms.ToTensor(),
        ]
    )

    pmnist_train_classifier = PathMNIST(split='train', download=False, size=28, root='data/pathmnist/', transform=transforms_classifier)

    pmnist_loader_classifier = DataLoader(
        pmnist_train_classifier,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=N_CPU_WORKERS,
    )

    eval_dict = model_eval(
        model,
        pmnist_train,
        pmnist_test,
        pmnist_loader_classifier,
        n_classes=9,
        eval_aug=False
    )

    with open(f'results/model_eval/{MODEL_FILENAME}_frac{frac}_eval.pkl', 'wb') as f:
        pickle.dump(eval_dict, f)

    ###################### ROTATION EVALUATION #########################
    from rotation_eval import pred_consistency_90deg, check_equivariance_torch

    nn.Module.check_equivariance_torch = check_equivariance_torch

    input_sizes = {}

    def hook_fn(name):
        def hook(module, inp, out):
            # Store input shape of this layer
            input_sizes[name] = tuple(inp[0].shape)
        return hook

    hooks = []
    for name, module in model.named_modules():
        hooks.append(module.register_forward_hook(hook_fn(name)))

    x = torch.randn(1, 3, 33, 33).to(device) 

    with torch.no_grad():
        model(x)

    for h in hooks:
        h.remove()
        
    # Equivariance error per layer
    eq_errors = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            error = module.check_equivariance_torch()
            eq_errors[name] = error

    # Classification consistency under rotation
    consistency = pred_consistency_90deg(model, pmnist_train, pmnist_test, n_classes=9)

    rotations_eval = {
        "Eq_errors": eq_errors,
        "Pred_consistency": consistency,
    }

    with open(f'results/model_eval_rotations/{MODEL_FILENAME}_frac{frac}_rotation_eval.pkl', 'wb') as f:
        pickle.dump(rotations_eval, f)

    print(f"Evaluation results saved to {MODEL_FILENAME}_frac{frac}_rotation_eval.pkl")