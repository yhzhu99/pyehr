from torch import nn


class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, act_layer=nn.GELU, drop=0.0, **kwargs):
        super(GRU, self).__init__()

        # hyperparameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.proj = nn.Linear(input_dim, hidden_dim)
        self.act = act_layer()
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

    def forward(self, x, **kwargs):
        x = self.proj(x)
        x, _ = self.gru(x)
        return x