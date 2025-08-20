import torch.nn as nn
import torch.nn.functional as F

class SeqAutoencoder(nn.Module):
    def __init__(self, emb_dim=512, hidden_dim=256, k=1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(emb_dim * k, hidden_dim),  # flatten [k*512] → hidden
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),   # another hidden layer
            nn.ReLU()
        )
        # maps hidden back to one embedding (512-dim prediction)
        self.decoder = nn.Linear(hidden_dim, emb_dim)

    def forward(self, x):
        # x shape: [batch, k, emb_dim]
        x = x.view(x.size(0), -1)   # flatten → [batch, k*emb_dim]
        h = self.encoder(x)         # compressed representation
        return self.decoder(h)      # predicted next embedding

def cosine_loss(pred, target):
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)

    # Cosine loss = 1 - cos(pred, target)
    return 1 - (pred * target).sum(dim=-1).mean()
