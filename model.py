import torch
import torch.nn as nn
import torch.nn.functional as F

from s2cnn import s2_equatorial_grid, S2Convolution, so3_equatorial_grid, SO3Convolution, so3_integrate, soft
import hyper


class Model(nn.Module):
    def __init__(self, nclasses):
        super().__init__()

        self.features = [hyper.R_IN, 40, 40, nclasses]
        self.bandwidths = [hyper.BANDWIDTH_IN, 32, 32, hyper.BANDWIDTH_OUT]
        self.linear1 = nn.Linear(nclasses + hyper.N_CATS, 50)
        self.linear2 = nn.Linear(50, 50)

        sequence = []

        # S2 layer
        grid = s2_equatorial_grid(max_beta=0, n_alpha=2 * self.bandwidths[0], n_beta=1)
        sequence.append(S2Convolution(self.features[0], self.features[1], self.bandwidths[0], self.bandwidths[1], grid))

        # SO3 layers
        for l in range(1, len(self.features) - 1):
            nfeature_in = self.features[l]
            nfeature_out = self.features[l + 1]
            b_in = self.bandwidths[l]
            b_out = self.bandwidths[l + 1]

            sequence.append(nn.BatchNorm3d(nfeature_in, affine=True))
            sequence.append(nn.ReLU())
            grid = so3_equatorial_grid(max_beta=0, max_gamma=0, n_alpha=2 * b_in, n_beta=1, n_gamma=1)
            sequence.append(SO3Convolution(nfeature_in, nfeature_out, b_in, b_out, grid))

        sequence.append(nn.BatchNorm3d(self.features[-1], affine=True))
        sequence.append(nn.ReLU())

        self.sequential = nn.Sequential(*sequence)

    def forward(self, x, target_index, cat_onehot):  # pylint: disable=W0221
        # concat after SO3 conv
        # B * C * a * b * c
        x = self.sequential(x)  # [batch, feature, beta, alpha, gamma]

        # B * C * N * 1 * 1
        features = F.grid_sample(x, target_index[:, :, None, None, :])
        # B * N * C
        features = features.squeeze(3).squeeze(3).permute([0, 2, 1]).contiguous()

        # B * N * (C + 16)
        prediction = torch.cat([features, cat_onehot[:, None, :].repeat(1, features.size(1), 1)], dim=2)

        # B * N * C
        prediction = F.relu(self.linear1(prediction))
        prediction = self.linear2(prediction)

        prediction = F.log_softmax(prediction, dim=2)
        return prediction
