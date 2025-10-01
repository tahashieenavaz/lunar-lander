import torch


class QNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = 256
        self.phi = torch.nn.Sequential(
            torch.nn.Linear(8, self.dim),
            torch.nn.LayerNorm(self.dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.dim, self.dim),
            torch.nn.LayerNorm(self.dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.dim, self.dim),
            torch.nn.LayerNorm(self.dim),
            torch.nn.ReLU(),
        )
        self.fc = torch.nn.Linear(self.dim, 4)

    def forward(self, x):
        features = self.phi(x)
        return self.fc(features)
