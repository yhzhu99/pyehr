from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, act_layer=nn.GELU, drop=0.0, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.proj = nn.Linear(input_dim, hidden_dim)
        self.act = act_layer()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            self.act,
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

    def forward(self, x, **kwargs):
        x = self.proj(x)
        x = self.mlp(x)
        return x