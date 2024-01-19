import os

class Config_Generative_Model:
    seed = 2022
    output_path = '/workspace/eeg_reconstruction/daekun/mamba-test/saves'
    eeg_signals_path = os.path.join('/workspace/eeg_reconstruction/shared/brain2image/eeg_5_95_std.pth')
    splits_path = os.path.join('/workspace/eeg_reconstruction/shared/brain2image/block_splits_by_image_single.pth')
        
    # Model Fields
    crop_ratio = 0.2
    img_size = 512
    eeg_len = 512
    subject = 4
    input_channels = 128

    # Trainer params
    batch_size_train: int = 512
    batch_size_eval: int = 300
    num_workers: int = 8
    max_train_steps: int = 1000

    # Optimizer
    adam_lr = 5e-4
    adam_betas = (0.9, 0.999)
    adam_eps = 1e-08
    adam_weight_decay = 0

    # Logging
    log_every_n_steps = 5

