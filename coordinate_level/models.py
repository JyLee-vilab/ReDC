import torch
import torch.nn as nn

class CR(nn.Module):
    """
    Confidence Re-encoder (CR).

    This module predicts the magnitude of the coordinate-wise
    calibration outputs from the detector confidence and bounding box
    features. A separate prediction head is used for each bounding-box
    coordinate.

    Arguments:
        D (int) : input feature dimension.

    Returns:
        None
    """
    def __init__(self, D):
        super().__init__()
        
        self.shift = nn.Parameter(torch.zeros(4))
        self.scale = nn.Parameter(torch.ones(4))

        self.feature_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(D-1, 2),
                nn.GELU(),
                nn.Linear(2,2),
                nn.GELU(),
                nn.Linear(2,1)
            )
            for _ in range(4)
        ])

    def forward(self, x):
        """
        Computes the coordinate-wise confidence logits.

        The detector confidence logit is combined with feature-dependent
        transformations to produce one confidence logit for each bounding-
        box coordinate.

        Arguments:
            x (torch.Tensor) : input feature tensor of shape (N, D)

        Returns:
            torch.Tensor : coordinate-wise confidence logits of shape
                        (N, 4).
        """
        value_outputs = []

        inverse_score = x[:, 0:1]
        other = x[:, 1:]

        for i, head in enumerate(self.feature_heads):

            out = head(other)
            shift_i = self.shift[i].view(1,1)
            value_logit = (inverse_score / torch.abs(out)) + shift_i

            value_outputs.append(value_logit)

        value_logits = torch.cat(value_outputs, dim=1)


        return value_logits
            
    def predict(self, x):
        """
        Predicts the coordinate-wise confidence magnitudes.

        Arguments:
            x (torch.Tensor or np.ndarray) : input feature tensor

        Returns:
            torch.Tensor : coordinate-wise confidence values in the
                        range [0, 1].
        """
        self.eval()

        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        with torch.no_grad():

            logits = self.forward(x)

            mag = torch.sigmoid(logits)

        return mag

class DDE(nn.Module):
    """
    Directional Displacement Estimator (DDE).

    This module predicts the direction of the localization error for
    each bounding-box coordinate. The predicted directions are later
    combined with the confidence magnitudes produced by the Confidence
    Re-encoder (CR).

    Arguments:
        D (int) : input feature dimension.

    Returns:
        None
    """
    def __init__(self, D):
        super().__init__()
        hidden_dim = 8 if D <= 16 else 16

        self.heads = nn.Sequential(
            nn.Linear(D, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4)
        )

    def forward(self, x):
        """
        Computes the directional prediction logits.

        Arguments:
            x (torch.Tensor) : input feature tensor of shape (N, D)

        Returns:
            torch.Tensor : directional logits for the four bounding-box
                        coordinates.
        """
        return self.heads(x)
    
    def predict(self, x, thr):
        """
        Predicts the direction of the localization error.

        The predicted probabilities are thresholded independently for each
        coordinate to obtain the final direction labels.

        Arguments:
            x (torch.Tensor or np.ndarray) : input feature tensor
            thr (array-like)               : decision thresholds for the
                                            four coordinates

        Returns:
            torch.Tensor : predicted coordinate directions encoded as
                        {-1, +1}.
        """
        self.eval()

        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        x = x.to(next(self.parameters()).device)

        with torch.no_grad():

            logits = self.forward(x)
            prob = torch.sigmoid(logits)
            thresh = torch.tensor(thr, device=prob.device, dtype=prob.dtype)
        

            sign = torch.where(prob > thresh, 1.0, -1.0)

            return sign