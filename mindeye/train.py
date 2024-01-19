import torch
from mindeye.dataset import create_EEG_dataset
from mindeye.classifier import MindEyeClassifier
from mindeye.config import Config_Generative_Model
import math
import torchvision.transforms as transforms
from einops import rearrange
import wandb
import os
# from accelerate import Accelerator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------
# CONFIGS & ACCELERATOR
# -----------------
config = Config_Generative_Model()
crop_pix = int(config.crop_ratio*config.img_size)
# accelerator = Accelerator()

# -----------------
# DATASET
# -----------------f
# Helper transforms
def normalize(img):
    if img.shape[-1] == 3:
        img = rearrange(img, 'h w c -> c h w')
    img = torch.tensor(img, dtype=torch.float32)
    img = img * 2.0 - 1.0 # to -1 ~ 1
    return img

class random_crop:
    def __init__(self, size, p):
        self.size = size
        self.p = p
    def __call__(self, img):
        if torch.rand(1) < self.p:
            return transforms.RandomCrop(size=(self.size, self.size))(img)
        return img
    
def channel_last(img):
        if img.shape[-1] == 3:
            return img
        return rearrange(img, 'c h w -> h w c')

# Define image transforms
img_transform_train = transforms.Compose([
    normalize,

    transforms.Resize((512, 512)),
    random_crop(config.img_size-crop_pix, p=0.5),

    transforms.Resize((512, 512)),
    channel_last
])
img_transform_test = transforms.Compose([
    normalize, 

    transforms.Resize((512, 512)),
    channel_last
])

eeg_latents_dataset_train, eeg_latents_dataset_test = create_EEG_dataset(eeg_signals_path = config.eeg_signals_path, splits_path = config.splits_path, 
            image_transform=[img_transform_train, img_transform_test], subject = config.subject)

training_loader = torch.utils.data.DataLoader(eeg_latents_dataset_train, batch_size=config.batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(eeg_latents_dataset_test, batch_size=config.batch_size_eval, shuffle=False)

# -----------------
# MODEL
# -----------------
print("Loading model:")
model = MindEyeClassifier(config=config)
model = model.to(device)

# -----------------
# OPTIMIZER
# -----------------
optimizer = torch.optim.AdamW(
    params=model.parameters(),
    lr=config.adam_lr,
    betas=config.adam_betas,
    eps=config.adam_eps,
    weight_decay=config.adam_weight_decay,
)



# -----------------
# LOGGING
# -----------------
config_dict = {attr: getattr(config, attr) for attr in dir(config) if not attr.startswith("__") and not callable(getattr(config, attr))}
wandb.init(
    # Set the project where this run will be logged
    project="mamba-sanity",
    entity= "alljoined1",
    # Track hyperparameters and run metadata
    config=config_dict,
)

# -----------------
# TRAINING
# -----------------
step_count = 0
train_loss = 0.0
best_val = 100.0

num_steps_per_epoch = math.ceil(len(training_loader))
# Afterwards we recalculate our number of training epochs
num_train_epochs = math.ceil(config.max_train_steps / num_steps_per_epoch)

# Multi-way classification
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(num_train_epochs):
    print(f"Epoch {epoch + 1}/{num_train_epochs}")
    # TRAIN LOOP
    model.train()

    for batch_index, train_batch in enumerate(training_loader):
        train_batch = {k: v.to(device) for k, v in train_batch.items()}

        print("Step:", batch_index, end="\r")
        optimizer.zero_grad()
        y_pred, loss_val = model(train_batch, criterion)
        loss_val.backward()
        train_loss += loss_val.item()  # do not track trace for this
        optimizer.step()

        step_count += 1
        if step_count % config.log_every_n_steps == 0:
            train_loss /= config.log_every_n_steps
            wandb.log({"train.loss": train_loss,})
            train_loss = 0.0
        if step_count >= config.max_train_steps:
            break     

    # VALIDATION LOOP
    with torch.no_grad():
        model.eval()
        val_loss = 0.0

        for val_batch in test_loader:
            val_batch = {k: v.to(device) for k, v in val_batch.items()}

            # Get validation loss
            y_pred, loss_val = model(val_batch, criterion)
            val_loss += loss_val.item()

        val_loss /= len(test_loader)
        wandb.log({"val.loss": val_loss,})

        # Save model if we are better than the best model so far
        if val_loss < best_val:
            print("New best model found! Saving..")
            torch.save(
                model.state_dict(),
                os.path.join(config.output_path, "best_contrastive.pt"),
            )
            best_val = val_loss
