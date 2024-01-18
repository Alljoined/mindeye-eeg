import kornia
import lightning.pytorch as pl
import pydantic
import torch
import torch.nn.functional as F
from kornia.augmentation.container import AugmentationSequential

from mindeye import utils
from mindeye.encoder import MambaEncoder
from mindeye.models import (
    BrainDiffusionPrior,
    Clipper,
    VersatileDiffusionPriorNetwork,
)


class LitMindEyeConfig(pydantic.BaseModel):
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


class LitMindEye(pl.LightningModule):

    def __init__(self, config):
        super().__init__()

        self.save_hyperparameters()

        self.full_config = config
        self.config: LitMindEyeConfig = LitMindEyeConfig.parse_obj(config)
        cfg = self.config

        if cfg.augment_images:
            self.augment = AugmentationSequential(
                kornia.augmentation.RandomResizedCrop((224, 224), (0.6, 1), p=0.3),
                kornia.augmentation.Resize((224, 224)),
                kornia.augmentation.RandomHorizontalFlip(p=0.5),
                kornia.augmentation.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.2,
                    hue=0.1, p=0.3,
                ),
                kornia.augmentation.RandomGrayscale(p=0.3),
                data_keys=["input"],
            )

        self.clipper = Clipper(
            "ViT-L/14",
            hidden_state=True,
            norm_embs=True,
        )
        clip_features = 768

        prior_network = VersatileDiffusionPriorNetwork(
            dim=clip_features,
            depth=6,
            dim_head=64,
            heads=(clip_features // 64),
            causal=False,
            num_tokens=257,
            learned_query_mode="pos_emb"
        )

        eeg2clip = MambaEncoder(
            input_features=cfg.input_channels,
            out_length=257,
            out_features=clip_features,
        )

        self.guidance_scale = 3.5
        self.timesteps = 100
        self.prior = BrainDiffusionPrior(
            net=prior_network,
            image_embed_dim=clip_features,
            condition_on_text_encodings=False,
            timesteps=self.timesteps,
            cond_drop_prob=0.2,
            image_embed_scale=None,
            eeg2clip=eeg2clip,
        )

        self.mixco_end = int(0.33 * cfg.max_epochs)
        soft_loss_temps = utils.cosine_anneal(0.004, 0.0075, cfg.max_epochs - self.mixco_end)
        self.register_buffer("soft_loss_temps", soft_loss_temps)

    def configure_optimizers(self):
        cfg = self.config

        params_wd = []
        params_no_wd = []

        blacklist = {"bias", "LayerNorm.bias", "LayerNorm.weight"}
        for M in [self.prior.net, self.prior.eeg2clip]:
            for n, p in M.named_parameters():
                if any((k in n) for k in blacklist):
                    params_no_wd.append(p)
                else:
                    params_wd.append(p)

        optimizer = torch.optim.AdamW(
            params=[
                {"params": params_wd, "weight_decay": 1e-2},
                {"params": params_no_wd, "weight_decay": 0.0},
            ],
            lr=cfg.lr,
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.lr,
            total_steps=int(cfg.max_epochs * cfg.steps_per_epoch),
            final_div_factor=1000,
            last_epoch=-1, pct_start=(2 / cfg.max_epochs)
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            }
        }

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, split="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, split="val")

    def _step(self, batch, batch_idx, split):
        cfg = self.config
        eeg, image = batch["eeg"], batch["image"]

        if cfg.augment_images:
            image = self.augment(image)

        mixco_enabled = (self.current_epoch < self.mixco_end)
        if mixco_enabled and (split == "train"):
            eeg, perm, betas, select = utils.mixco(eeg)
        else:
            perm = betas = select = None

        print(image.dtype, image.shape)
        exit()
        clip_target = self.clipper.embed_image(image).float()
        clip_inputs, clip_inputs_proj = self.prior.eeg2clip(eeg)

        loss_prior, aligned_clip_inputs = self.prior(text_embed=clip_inputs, image_embed=clip_target)
        aligned_clip_inputs = aligned_clip_inputs / self.prior.image_embed_scale

        clip_inputs_norm = F.normalize(clip_inputs_proj.flatten(1), dim=-1)
        clip_target_norm = F.normalize(clip_target.flatten(1), dim=-1)

        if mixco_enabled:
            loss_nce = utils.mixco_nce(
                clip_inputs_norm,
                clip_target_norm,
                temp=.006,
                perm=perm, betas=betas, select=select,
            )
        else:
            epoch_temp = self.soft_loss_temps[self.current_epoch - self.mixco_end]
            loss_nce = utils.soft_clip_loss(
                clip_inputs_norm,
                clip_target_norm,
                temp=epoch_temp,
            )

        loss = loss_nce + (30 * loss_prior)
        assert loss.isfinite().all()

        with torch.no_grad():
            sims_base = F.cosine_similarity(clip_target_norm, clip_inputs_norm).mean().item()

            # Forward and backward top 1 accuracy
            labels = torch.arange(len(clip_target_norm)).to(self.device)
            fwd_percent_correct = utils.topk(
                utils.batchwise_cosine_similarity(clip_inputs_norm, clip_target_norm), labels, k=1)
            bwd_percent_correct = utils.topk(
                utils.batchwise_cosine_similarity(clip_target_norm, clip_inputs_norm), labels, k=1)

        # Logging
        log_kwargs = dict(batch_size=image.shape[0], sync_dist=(split != "train"))
        self.log(f"{split}/loss_nce", loss_nce, **log_kwargs)
        self.log(f"{split}/loss_prior", loss_prior, **log_kwargs)
        self.log(f"{split}/loss", loss, **log_kwargs)

        self.log(f"{split}/sims_base", sims_base, **log_kwargs)
        self.log(f"{split}/fwd_percent_correct", fwd_percent_correct, **log_kwargs)
        self.log(f"{split}/bwd_percent_correct", bwd_percent_correct, **log_kwargs)

        return loss
