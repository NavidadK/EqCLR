import torch
import numpy as np
from escnn.nn import GeometricTensor
from escnn.nn import EquivariantModule
import torch.nn as nn
from evaluation import dataset_to_X_y, lin_eval_rep
import torchvision.transforms.functional as TF
import tensorflow as tf
from torch.utils.data import DataLoader
from IPython.display import display
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_module_device(module):
    # Check parameters
    for p in module.parameters():
        return p.device
    # Check buffers (ESCNN layers often store filters as buffers)
    for b in module.buffers():
        return b.device
    return torch.device("cpu")

############ Equivariance checks ############

def check_equivariance_90deg(self, atol: float = 1e-7, rtol: float = 1e-5, x_size=11):
    """
    Check equivariance for 90-degree rotations.
    Automatically uses the same device as the module.
    """

    device = get_module_device(self)

    # Input tensor
    c = self.in_type.size
    
    x = torch.randn(3, c, *[x_size]*self.in_type.gspace.dimensionality, device=device) # random input
    x = GeometricTensor(x, self.in_type)

    errors = {}
    group_elements = self.in_type.gspace.fibergroup.elements
    testing_elements = [g for g in group_elements if np.isclose(g.to('radians') % (np.pi/2), 0)]

    for el in testing_elements:
        # Forward + transform
        out1 = self(x).transform(el).tensor.detach().cpu().numpy()
        out2 = self(x.transform(el)).tensor.detach().cpu().numpy()

        # Compute absolute error
        errs = np.abs(out1 - out2).reshape(-1)

        if not np.allclose(out1, out2, atol=atol, rtol=rtol):
            print(f"⚠ WARNING: Error for element {el}: "
                  f"max={errs.max():.6f}, mean={errs.mean():.6f}, var={errs.var():.6f}")
        
        #errors.append((el, errs.mean()))
        errors[str(el)] = errs.mean()

    return errors

EquivariantModule.check_equivariance_90deg = check_equivariance_90deg

def check_equivariance_torch(self, atol: float = 1e-7, rtol: float = 1e-5, x='odd'):
    """
    Check equivariance for 90-degree rotations.
    Automatically uses the same device as the module.
    """

    params = list(self.parameters())
    device = params[0].device if len(params) > 0 else torch.device('cpu')
    
    if isinstance(self, nn.Conv2d):
        if x == 'odd':
            size = (3, self.in_channels, 11, 11) # odd size
        elif x == 'even':
            size = (3, self.in_channels, 10, 10) # even size 
        # Random input
        x = torch.randn(*size, device=device) # random input

        errors = {}
        
        # Rotations to test (multiples of 90°)
        for k, deg in enumerate([0, 90, 180, 270]):
            # Rotate input by k*90 degrees (on last two dims H,W)
            x_rot = torch.rot90(x, k=k, dims=(2,3))
            
            out1 = torch.rot90(self(x).detach().cpu(), k=k, dims=(2,3)).cpu().numpy()
            out2 = self(x_rot).detach().cpu().numpy()

            # Compute error
            errs = np.abs(out1 - out2).reshape(-1)
            errors[deg] = errs.mean()
            
            if not np.allclose(out1, out2, atol=atol, rtol=rtol):
                print(f"⚠ WARNING: Error for element {k}: "
                    f"max={errs.max():.6f}, mean={errs.mean():.6f}, var={errs.var():.6f}")
        return errors
    else:
        pass

nn.Module.check_equivariance_torch = check_equivariance_torch


############### Prediction consistency under rotation ###############

# Not used atm
def rotate_90(x, angle):
    """Rotate a BCHW PyTorch tensor by 0/90/180/270 degrees without interpolation."""
    if angle == 0:
        return x
    elif angle == 90:
        return x.transpose(-1, -2).flip(-1)        # rotate 90° CCW
    elif angle == 180:
        return x.flip(-1).flip(-2)
    elif angle == 270:
        return x.transpose(-1, -2).flip(-2)        # rotate 90° CW
    else:
        raise ValueError("Angle must be 0, 90, 180, or 270")


def pred_consistency_90deg(model, data_train, data_test, n_classes):

    ########### classifier #################
    with torch.no_grad():
            X_train, y_train, Z_train = dataset_to_X_y(data_train, model)
            X_test, y_test, Z_test = dataset_to_X_y(data_test, model)

    acc, classifier = lin_eval_rep(X_train, y_train, X_test, y_test, n_classes=n_classes, n_epochs=10, return_model=True)

    ########### rotation consistency #################

    with torch.no_grad():
        pred_per_angle = {}
        for angle in [0, 90, 180, 270]:
            y_hat = []

            for batch_idx, batch in enumerate(DataLoader(data_test, batch_size=1024, shuffle=False)):
                images, labels = batch
                rotated_images = []

                # rotate images
                for img in images:
                    img_rot = TF.rotate(img, angle, expand=False)
                    rotated_images.append(img_rot)
                rotated_images = torch.stack(rotated_images).to(device)

                # forward pass
                h, z = model(rotated_images)
                # classification
                logits = classifier(h)
                preds = logits.argmax(dim=1).cpu().numpy()
                y_hat.append(preds)
            
            y_hat = np.hstack(y_hat)
            pred_per_angle[angle] = y_hat

        # df of predictions
        y_hat_matrix = np.vstack([pred_per_angle[angle] for angle in [0, 90, 180, 270]]).T  # shape (n_samples, 4)
        num_unique_preds = np.array([len(np.unique(p)) for p in y_hat_matrix])
        # df_y_hat = pd.DataFrame(y_hat_matrix, columns=['pred_0', 'pred_90', 'pred_180', 'pred_270'])
        # df_y_hat['num_unique_preds'] = num_unique_preds
        # display(df_y_hat)

        # Compute percentages
        total_samples = len(num_unique_preds)
        print(f'total_samples: {total_samples}')
        percentages = {
            'all_same': (num_unique_preds == 1).sum() / total_samples * 100,
            '1 different': (num_unique_preds == 2).sum() / total_samples * 100,
            '2 different': (num_unique_preds == 3).sum() / total_samples * 100,
            'all different': (num_unique_preds == 4).sum() / total_samples * 100
        }

        for k, v in percentages.items():
            print(f"{k}: {v:.2f}%")

    return percentages
            
