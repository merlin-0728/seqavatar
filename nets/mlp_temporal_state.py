import torch.nn as nn


class TemporalStateEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=64, num_layers=3):
        super().__init__()

        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            dilation = 2 ** i
            layers += [
                nn.Conv1d(
                    in_channels=in_dim,
                    out_channels=hidden_dim,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation,
                ),
                nn.ReLU(),
            ]
            in_dim = hidden_dim

        self.tcn = nn.Sequential(*layers)
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, state_seq):
        # state_seq: [B, L, C]
        x = state_seq.permute(0, 2, 1)
        feat = self.tcn(x)
        feat_t = feat[:, :, -1]
        return self.out(feat_t)
