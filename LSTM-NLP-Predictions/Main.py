from random import random

import pandas as pd
import numpy as np
import torch

from .BpeTokenizer import BpeTokenizer
from .Model import Model
from .MyDataset import MyDataset
from .Collator import Collator
from .Trainer import Trainer
from .generation import generate


df = pd.read_csv('dataset.csv')
train_texts = df['text'][:-1024].tolist()
eval_texts = df['text'][-1024:].tolist()

tokenizer = BpeTokenizer()
tokenizer.train(train_texts[:2048], max_vocab=2048)

train_dataset = MyDataset(train_texts, tokenizer, max_length=128)
eval_dataset = MyDataset(eval_texts, tokenizer, max_length=128)
collator = Collator(tokenizer.pad_token_id)

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

model = Model(tokenizer.get_vocab_size(), emb_size=128, hidden_size=256, num_layers=2, dropout=0.1)

trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        n_epochs=8,
        train_batch_size=32,
        eval_batch_size=32,
        eval_steps=64,
        collator=collator,
        lr=1e-2,
        ignore_index=tokenizer.pad_token_id
)

trainer.train()
#Получим жадную генерацию
generate(model, tokenizer, temperature=0.5, top_k=20)
