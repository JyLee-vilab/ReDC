import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import random
import numpy as np

from coordinate_level.models import CR, DDE

def Train_CR(train_input, train_acc, lr=1e-1, epochs=100, batch_size=64, device=None, seed=1234):
    """
    Trains the Confidence Re-encoder (CR).

    This function learns a mapping from the detector outputs to
    coordinate-wise confidence values using binary cross-entropy loss.
    The target values are the magnitudes of the Coordinate-wise Alignment
    Ratios (CARs).

    Arguments:
        train_input (torch.Tensor or np.ndarray)
                                : input features used for training
                                    the CR
        train_acc (torch.Tensor or np.ndarray)
                                : target coordinate-wise CAR values
        lr (float)                 : learning rate
        epochs (int)               : number of training epochs
        batch_size (int)           : mini-batch size
        device (str or None)       : computation device. If None,
                                    CUDA is used when available.
        seed (int)                 : random seed for reproducibility

    Returns:
        CR : trained Confidence Re-encoder model.

    Note:
        The model is trained to predict the magnitude of the
        coordinate-wise calibration targets by minimizing binary
        cross-entropy loss between the predicted confidence and the
        absolute CAR values.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if not isinstance(train_input, torch.Tensor):
        input_tensor = torch.tensor(train_input, dtype=torch.float32)
    else:
        input_tensor = train_input.to(torch.float32)

    y_tensor = torch.tensor(train_acc, dtype=torch.float32)

    input_tensor = input_tensor.to(device)
    y_tensor = y_tensor.to(device)

    N, D = input_tensor.shape

    dataset = TensorDataset(input_tensor, y_tensor)
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model = CR(D).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):

        model.train()

        for xb, yb in loader:

            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            logits = model(xb)

            value_target = torch.abs(yb)

            value = torch.sigmoid(logits)

            loss = F.binary_cross_entropy(value, value_target)

            loss.backward()
            optimizer.step()

    return model


def Train_DDE(train_input, train_acc, train_car=None, car_cut=0.9, lr=1e-3, epochs=100, batch_size=128, device=None, seed=1234):
    """
    Trains the Directional Displacement Estimator (DDE).

    This function learns the direction of the localization error for
    each bounding-box coordinate using binary classification. Samples
    whose Coordinate Alignment Ratio (CAR) exceeds a specified
    threshold can optionally be excluded from training, allowing the
    model to focus on detections with meaningful localization errors.

    Arguments:
        train_input (torch.Tensor or np.ndarray)
                                : input features used for training
                                    the DDE
        train_acc (torch.Tensor or np.ndarray)
                                : target coordinate directions
        train_car (torch.Tensor or np.ndarray, optional)
                                : coordinate-wise CAR values used
                                    for sample selection
        car_cut (float)            : CAR threshold for selecting
                                    training samples
        lr (float)                 : learning rate
        epochs (int)               : number of training epochs
        batch_size (int)           : mini-batch size
        device (str or None)       : computation device. If None,
                                    CUDA is used when available.
        seed (int)                 : random seed for reproducibility

    Returns:
        DDE : trained Directional Displacement Estimator model.

    Note:
        Training samples whose absolute CAR is greater than or equal to
        `car_cut` are ignored when `train_car` is provided. The model is
        optimized using binary cross-entropy with logits loss over the
        valid coordinate directions only.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if not isinstance(train_input, torch.Tensor):
        input_tensor = torch.tensor(train_input, dtype=torch.float32)
    else:
        input_tensor = train_input.to(torch.float32)

    if not isinstance(train_acc, torch.Tensor):
        y_tensor = torch.tensor(train_acc, dtype=torch.float32)
    else:
        y_tensor = train_acc.to(torch.float32)

    if train_car is not None:
        if not isinstance(train_car, torch.Tensor):
            car_tensor = torch.tensor(train_car, dtype=torch.float32)
        else:
            car_tensor = train_car.to(torch.float32)
    else:
        car_tensor = torch.zeros_like(y_tensor)

    input_tensor = input_tensor.to(device)
    y_tensor = y_tensor.to(device)
    car_tensor = car_tensor.to(device)

    N, D = input_tensor.shape

    if y_tensor.shape[0] != N:
        raise ValueError(f"train_input N={N}, train_acc N={y_tensor.shape[0]}")

    if car_tensor.shape[0] != N:
        raise ValueError(f"train_input N={N}, train_car N={car_tensor.shape[0]}")

    dataset = TensorDataset(input_tensor, y_tensor, car_tensor)

    g = torch.Generator()
    g.manual_seed(seed)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False, generator=g)

    model = DDE(D).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(epochs):

        model.train()
        
        for xb, yb, db in loader:

            xb = xb.to(device)
            yb = yb.to(device)
            db = db.to(device)

            if train_car is not None:
                car_mask = (torch.abs(db) < car_cut)
            else:
                car_mask = torch.ones_like(yb, dtype=torch.bool)

            optimizer.zero_grad()

            logits = model(xb)
            sign_target = (yb > 0).float()

            label_mask = (yb != 0)

            valid_mask = label_mask & car_mask

            if valid_mask.sum() == 0:
                continue

            bce_loss = F.binary_cross_entropy_with_logits(logits, sign_target, reduction='none')

            loss = bce_loss[valid_mask].mean()

            loss.backward()
            optimizer.step()

            
    return model