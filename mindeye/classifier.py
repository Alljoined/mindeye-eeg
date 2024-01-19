import torch.nn as nn
from mindeye.encoder import MambaEncoder
import pydantic


class MindEyeConfig(pydantic.BaseModel):
    #
    # Set by datamodule
    input_channels: int = -1
    steps_per_epoch: int = -1

    # ===============
    # Training Fields
    # ===============

    augment_images: bool = True

    max_epochs: int = 240
    lr: float = 3e-4

    sample_seed: int = 42



class MindEyeClassifier(nn.Module):
    def __init__(self, config):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()

        clip_features = 768
        self.eeg2clip = MambaEncoder(
            input_features=config.eeg_len,
            out_length=257,
            out_features=clip_features,
            hidden_features=256,
            depth=8
        )

        # self.flatten = nn.Flatten()
        # self.avgpool = nn.Avg
        self.fc1 = nn.Linear(clip_features, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 40)

    def forward(self, batch, critereon):
        eeg = batch["eeg"]
        y = batch["label"]

        x = self.eeg2clip(eeg)[0]
        x = x.mean(dim=1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        
        loss = critereon(x, y)

        return x, loss