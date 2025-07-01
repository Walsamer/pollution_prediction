import torch
import torch.nn as nn


def choose_device() -> torch.device:
    """
    Selects the best available device for PyTorch computations.

    Returns
    -------
    torch.device
        'cuda' if available, else 'mps' if available, else 'cpu'.
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class PMModel(nn.Module):
    """
    A fully connected ffn model for regression of PM2.5

    Parameters:
      input_dim : int
          Number of input features.
      hidden_size : list[int]
          Sizes of each hidden layer. The length of this list defines the number of hidden layers.
      dropout : float
          Dropout probability applied after each hidden layer.
    """
    def __init__(self, input_dim: int, hidden_size: list[int], dropout: float = 0.0):
        super().__init__()
        layers = []  # empty, will be overwritten
        in_dim = input_dim
        for idx, hs in enumerate(hidden_size): #building the hiddenlayers
            layers.append(nn.Linear(in_dim, hs))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            in_dim = hs

        layers.append(nn.Linear(in_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Parameters
          x : torch.Tensor
              Input tensor of shape (batch_size, input_dim).
  
        Returns
          torch.Tensor
              Output tensor of shape (batch_size, 1) with raw regression predictions.
        """
        return self.net(x)
