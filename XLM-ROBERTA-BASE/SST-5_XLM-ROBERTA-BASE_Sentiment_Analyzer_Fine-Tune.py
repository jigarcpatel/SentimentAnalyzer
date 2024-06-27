#!/usr/bin/env python
# coding: utf-8

# In[21]:


#Step 1: Install Required Libraries
get_ipython().system('pip install transformers datasets torch pandas scikit-learn accelerate -U')
get_ipython().system('pip install numpy')
get_ipython().system('pip install sentencepiece')


# In[22]:


import os
import torch
import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, Trainer, TrainingArguments

# Disable WandB
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# In[23]:


# Load the dataset
dataset = load_dataset("SetFit/sst5")

# Convert the dataset to pandas DataFrames
train_df = pd.DataFrame(dataset['train'])
test_df = pd.DataFrame(dataset['test'])

# Check the column names
print(train_df.columns)

# Split the train_df into training and validation sets
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

# Reset index
train_df.reset_index(drop=True, inplace=True)
val_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)


# In[24]:


from transformers import AutoTokenizer

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

train_encodings = tokenizer(train_df['text'].tolist(), truncation=True, padding=True)
val_encodings = tokenizer(val_df['text'].tolist(), truncation=True, padding=True)
test_encodings = tokenizer(test_df['text'].tolist(), truncation=True, padding=True)

train_labels = train_df['label'].tolist()
val_labels = val_df['label'].tolist()
test_labels = test_df['label'].tolist()


# In[25]:


import torch

class SST5Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = SST5Dataset(train_encodings, train_labels)
val_dataset = SST5Dataset(val_encodings, val_labels)
test_dataset = SST5Dataset(test_encodings, test_labels)


# In[26]:


from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load the model
model = AutoModelForSequenceClassification.from_pretrained('xlm-roberta-base', num_labels=5)

# Define the metrics
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average='weighted')
    acc = accuracy_score(p.label_ids, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',          
    evaluation_strategy="epoch",     
    save_strategy="epoch",  # Make sure the evaluation and save strategy match
    save_total_limit=2,              
    num_train_epochs=10,             
    per_device_train_batch_size=16,  
    per_device_eval_batch_size=32,   
    warmup_steps=500,                
    weight_decay=0.01,               
    logging_dir='./logs',            
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    disable_tqdm=False,
    report_to=[]  # Disable wandb
)

# Initialize the Trainer
trainer = Trainer(
    model=model,                        
    args=training_args,                 
    train_dataset=train_dataset,         
    eval_dataset=val_dataset,             
    compute_metrics=compute_metrics,      
)

# Train the model
trainer.train()


# In[27]:


# Evaluate on the test set
results = trainer.evaluate(test_dataset)
print(results)


# In[ ]:





# In[ ]:





# In[ ]:





# In[28]:


#from huggingface_hub import notebook_login

#notebook_login()


# In[29]:


#trainer.push_to_hub()


# In[ ]:




