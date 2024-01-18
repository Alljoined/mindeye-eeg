from typing import Literal, Optional

import lightning.pytorch as pl
import pydantic_cli
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, ModelSummary
from lightning.pytorch.loggers import WandbLogger

from mindeye.dataset import EEGDataModule
from mindeye.lit import LitMindEye, LitMindEyeConfig
from mindeye.utils import DATA_ROOT, LOG_DIR, random_checkpoint_dir


class TrainConfig(LitMindEyeConfig):
    #
    seed: int = 100

    accelerator: str = "cpu"
    num_nodes: int = 1
    devices: int = 1
    strategy: Optional[str] = "auto"

    precision: Literal["32", "16-mixed", "bf16-mixed"] = "bf16-mixed"

    # =================
    # Datamodule Fields
    # =================

    eeg_path: str = str(DATA_ROOT / "eeg_5_95_std.pth")
    batch_size_train: int = 32
    batch_size_eval: int = 300
    num_workers: int = 8

    # ===============
    # Training Fields
    # ===============

    resume_path: Optional[str] = None

    # ==============
    # Logging Fields
    # ==============

    wandb: bool = False
    wandb_entity: Optional[str] = None
    wandb_project: str = "mindeye_eeg"

    checkpoint: bool = False
    checkpoint_dir: Optional[str] = None

    log_every_n_steps: int = 5
    progress_bar: bool = False

    class Config(pydantic_cli.DefaultConfig):
        extra = "forbid"
        CLI_BOOL_PREFIX = ("--enable_", "--disable_")


def train(config: TrainConfig):
    cfg = config

    # Seeding
    pl.seed_everything(cfg.seed, workers=True)

    # Load dataset
    data = EEGDataModule(
        eeg_path=cfg.eeg_path,
        batch_size=cfg.batch_size_train,
        eval_batch_size=cfg.batch_size_eval,
        num_workers=cfg.num_workers,
    )
    cfg.input_channels = data.input_channels
    cfg.steps_per_epoch = int(data.num_train // (cfg.devices * cfg.batch_size_train))

    # Initialize and load model
    model = LitMindEye(config=dict(cfg))

    # Initialize trainer
    callbacks = [ModelSummary(max_depth=2)]

    if cfg.wandb:
        logger = WandbLogger(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            log_model=False,
            save_dir=LOG_DIR,
        )
        callbacks.append(LearningRateMonitor())
    else:
        logger = False

    if cfg.checkpoint:
        if cfg.checkpoint_dir is None:  # set to some random unique folder
            cfg.checkpoint_dir = random_checkpoint_dir()
        callbacks.append(
            ModelCheckpoint(
                dirpath=cfg.checkpoint_dir,
                monitor="val/loss",
                mode="min",
                save_top_k=1,
                save_last=True,
                verbose=True,
            )
        )

    trainer = pl.Trainer(
        accelerator=cfg.accelerator,
        num_nodes=cfg.num_nodes,
        devices=cfg.devices,
        strategy=cfg.strategy,
        callbacks=callbacks,
        enable_checkpointing=cfg.checkpoint,
        logger=logger,
        max_epochs=cfg.max_epochs,
        log_every_n_steps=cfg.log_every_n_steps,
        enable_progress_bar=cfg.progress_bar,
        use_distributed_sampler=True,
    )

    # Start training
    trainer.fit(model=model, datamodule=data, ckpt_path=cfg.resume_path)

    return 0


if __name__ == "__main__":
    pydantic_cli.run_and_exit(TrainConfig, train)
