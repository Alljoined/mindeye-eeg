from typing import Any, Dict

import PIL.Image
import lightning.pytorch as pl
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoProcessor

from mindeye.utils import DATA_ROOT

IMAGENET_ROOT = DATA_ROOT / "imageNet_images"


class EEGDataset(Dataset):

    def __init__(self, eeg_path):
        super().__init__()

        # Load EEG signals
        data = torch.load(eeg_path)

        # We'll use only subject 4 for now 
        self.data = [x for x in data["dataset"] if (x["subject"] == 4)]
        self.images = data["images"]

        self.eeg_length = 440
        self.image_size = 512

        # Compute size
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example: Dict[str, Any] = self.data[idx]

        # Crop EEG sequence
        eeg = example["eeg"].float().T
        eeg = eeg[20:460, :]
        assert eeg.shape[0] == self.eeg_length

        # Get label
        label = torch.tensor(example["label"], dtype=torch.long)

        # Get image
        image_name = self.images[example["image"]]
        image_path = IMAGENET_ROOT / image_name.split("_")[0] / f"{image_name}.JPEG"
        image_raw = PIL.Image.open(image_path).convert("RGB")

        # Transforms
        image = self._scale_and_resize(image_raw)
        image_raw = self.processor(images=image_raw, return_tensors="pt")
        image_raw["pixel_values"] = image_raw["pixel_values"].squeeze(0)

        # Done!
        return {"eeg": eeg, "label": label, "image": image, "image_raw": image_raw}

    def _scale_and_resize(self, image):
        image = TF.pil_to_tensor(image)
        assert image.shape[0] == 3  # (3 H W)
        image = (2 * (image.float() / 255)) - 1  # [0, 255] -> [-1, 1]
        image = TF.resize(image, [self.image_size, self.image_size], antialias=True)
        return image


class EEGDataModule(pl.LightningDataModule):

    def __init__(
        self,
        eeg_path=str(DATA_ROOT / "eeg_5_95_std.pth"),
        batch_size=32,
        eval_batch_size=300,
        num_workers=2,
    ):
        super().__init__()

        self.batch_sizes = {"train": batch_size, "val": eval_batch_size}
        self.num_workers = num_workers

        # TODO: hacky load of splits
        split_indices = torch.load(DATA_ROOT / "block_splits_by_image_single.pth")
        split_indices = split_indices["splits"][0]

        self.base_dataset = EEGDataset(eeg_path)
        self.datasets = {
            split: Subset(self.base_dataset, indices=indices)
            for split, indices in split_indices.items()
        }

        self.input_channels = 128
        self.num_train = len(self.datasets["train"])

    def train_dataloader(self):
        return self._loader(split="train", shuffle=True, drop_last=True)

    def val_dataloader(self):
        return self._loader(split="val")

    def _loader(self, split, shuffle=False, drop_last=False):
        return DataLoader(
            dataset=self.datasets[split],
            batch_size=self.batch_sizes[split],
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=drop_last,
            pin_memory=True,
        )
