import argparse
import glob
import os
import json
import time
import logging
import random
import re
import copy
from itertools import chain
from tqdm.auto import tqdm

from dataset import WikiSqlDataset
from model import LoggingCallback,SeqGenSQL
import multiprocessing
import torch
import numpy as np
import pytorch_lightning as pl

######################################################################
## Utilities
######################################################################
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# main
logger = logging.getLogger(__name__)

num_of_workers = multiprocessing.cpu_count()

# suppress warning - Lightning 0.8.4 introduces an issue that could generate overwhelming warning messages
logging.basicConfig(level=logging.ERROR)

args_dict = dict(
    data_dir="data", # path for data files
    output_dir = ".",
    default_root_dir =".", # path to save the checkpoints
    model_name_or_path="t5-base",
    #tokenizer_name_or_path=base_model,
    max_seq_length= 512,
    max_output_length = 200,
    learning_rate=2e-4,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    warmup_steps=0,
    num_train_epochs=25,
    gradient_accumulation_steps=16,
    gpus = -1,
    early_stop_callback=False,
    max_grad_norm=1.0, # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
    seed=42,
    include_data_type = True,
    num_sample_rows = 3,
    data_aug = [],#  ['select_column', 'where_value']
    #   generated_data_files =["datagen/epoch_20/e20_1.jsonl"]
    generated_data_files = [],  
    use_modified_network = True, # this is added flag to identify if modified netowkr is used. True is modified network is used, False to use original T5
    #deterministic=True, #reproducibility, could make training slower
    #auto_scale_batch_size='binsearch',
    benchmark=True,
    num_of_workers = multiprocessing.cpu_count(),
)


args = argparse.Namespace(**args_dict)

if args.generated_data_files != []:
    args.data_aug = []
    
if isinstance(args.gpus, list):
    args.n_gpu= len(args.gpus)
elif args.gpus == -1:
    args.n_gpu = torch.cuda.device_count()

args.train_batch_size= 2 * args.n_gpu
args.eval_batch_size = 2 * args.n_gpu

seed_everything(args.seed)

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    filepath=os.path.join(args.output_dir, "base_gated_{epoch:02d}-{val_loss:.5f}"), prefix="", monitor="val_loss", mode="min", save_top_k=5
)

train_params = dict(
    accumulate_grad_batches=args.gradient_accumulation_steps,
    gpus=args.gpus,
    max_epochs=args.num_train_epochs,
    early_stop_callback=False,
    gradient_clip_val=args.max_grad_norm,
    checkpoint_callback=checkpoint_callback,
    callbacks=[LoggingCallback()],
)

if args.n_gpu > 1:
    train_params["distributed_backend"] = "dp"

#tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)

# initialize model
model = SeqGenSQL(args)

# restore full training state
# trainer = pl.Trainer(resume_from_checkpoint='t5_checkpoints/epoch=15.ckpt', gpus=1, )
# multi GPUs: 
#trainer = pl.Trainer(resume_from_checkpoint='t5_checkpoints/base_gated_e03_0.2470.ckpt', **train_params)

trainer = pl.Trainer(**train_params)

# Train
trainer.fit(model) 