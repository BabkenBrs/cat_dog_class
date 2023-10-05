import torch
from torch import nn


# a special module that converts [batch, channel, w, h] to [batch, units]: tf/keras style
class Flatten(nn.Module):
    def forward(self, x):
        # finally we have it in pytorch
        return torch.flatten(x, start_dim=1)


class Simple_Stupid_model(nn.Module):
    def __init__(self, input_shape=24 * 24, num_classes=2, input_channels=1):
        super().__init__()
        self.model = nn.Sequential()

        # reshape from "images" to flat vectors
        self.model.add_module("flatten", Flatten())
        # dense "head"
        self.model.add_module("dense1", nn.Linear(3 * 96 * 96, 256))
        self.model.add_module("dense3", nn.Linear(256, 128))
        # logits for NUM_CLASSES=2: cats and dogs
        self.model.add_module("dense4_logits", nn.Linear(128, 2))

    def forward(self, inp):
        out = self.model(inp)
        return out
