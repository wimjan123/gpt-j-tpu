#code to train gpt-j on google tpu-v4-8 

import os
import torch
import transformers
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, Trainer, TrainingArguments
from datasets import load_dataset

# Load the dataset
dataset = load_dataset('Dahoas/synthetic-instruct-gptj-pairwise', split='train')

# Load the GPT-J model and tokenizer
model = GPT2LMHeadModel.from_pretrained('EleutherAI/gpt-j-6B')
tokenizer = GPT2TokenizerFast.from_pretrained('EleutherAI/gpt-j-6B')

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'])

tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=['text'])

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy = 'steps',
    eval_steps = 500,
    save_steps = 500,
    num_train_epochs = 10,
    per_device_train_batch_size = 2,
    per_device_eval_batch_size = 2,
    learning_rate = 2e-5,
    weight_decay = 0.01,
    push_to_hub = False,
    logging_steps = 500,
)

# Define the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                                'attention_mask': torch.stack([f[1] for f in data]),
                                'labels': torch.stack([f[0] for f in data])},
)

# Train the model
trainer.train()

