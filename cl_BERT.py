import numpy as np
import random

from transformers import BertTokenizer,BertForSequenceClassification, AdamW
import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, TensorDataset, random_split
from transformers import get_linear_schedule_with_warmup

from evaluate import eval_BERT
from evaluate import f1_score_BERT


def save_model(model, filename, path=None):
    model.save(filename)

def tokenize(data, col_x, col_y,
             pad_maxlen,
             pretrained_model='bert-base-uncased'):
    tokenizer_bert = BertTokenizer.from_pretrained(pretrained_model, do_lower_case=True)

    data[col_y] = data[col_y].replace(4,1)
    sent = data[col_y].values
    text = data[col_x].values
    
    input_ids = []
    attention_masks = []
    for item in text:
        tokenizer_encoding = tokenizer_bert.encode_plus(
            item,
            max_length=pad_maxlen,
            add_special_tokens=True, #Add [CLS] and [SEP]
            padding='max_length', #padding=True: defaults to the longest sequence / padding='max_length': pad to max_length
            truncation=True, # truncate to max_length; implemented so that my program does not exceed the GPU memory
            return_attention_mask=True, #attention_mask: List of indices specifying which tokens should be attended to by the model
            return_tensors='pt' #'tf'/'pt'/'np': returns tensorflow, pytorch, np.ndarray objects
            )
        input_ids.append(tokenizer_encoding['input_ids'])
        attention_masks.append(tokenizer_encoding['attention_mask'])
        
    input_ids = torch.cat(input_ids,dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    sent = torch.tensor(sent)
        
    dataset_bert = TensorDataset(input_ids,attention_masks,sent)
    return dataset_bert


def load(dataset_bert, test_ratio, test_seed, batch_size = 16):
    # random_split(): randomly split a dataset into non-overlapping new datasets of given lengths
    train_size = int((1-test_ratio)*len(dataset_bert))
    test_size = len(dataset_bert)-train_size
    train_bert, test_bert = random_split(dataset_bert, [train_size, test_size], 
                                         generator=torch.Generator().manual_seed(test_seed))
    
    train_dl = DataLoader(train_bert, sampler = RandomSampler(train_bert), batch_size=batch_size)
    test_dl = DataLoader(test_bert, sampler= SequentialSampler(test_bert), batch_size=batch_size)
    
    print(f'Training size: {len(train_dl)}')
    print(f'Test size: {len(test_dl)}')

    return train_dl, test_dl

    
def finetune(train_dl, test_dl, 
             epochs, 
             lr = 2e-5, eps = 1e-6,
             valid_seed = None,
             pretrained_model='bert-base-uncased'):
    
    # BertForSequenceClassification: inherits from PreTrainedModel
    # It has a sequence classification/regression head on top (a linear layer on top of the pooled output) e.g. for GLUE tasks.
    model_bert = BertForSequenceClassification.from_pretrained(pretrained_model,
                                                               num_labels=2,
                                                               output_attentions=False,
                                                               output_hidden_states=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_bert.to(device)
    print(f'Device: {device}')
    
    if valid_seed:
        random.seed(valid_seed)
        np.random.seed(valid_seed)
        torch.manual_seed(valid_seed)
        torch.cuda.manual_seed_all(valid_seed)
    
    # The training loss is the sum of the mean masked LMlikelihood and the mean next sentence prediction likelihood.
    optimizer = AdamW(params=model_bert.parameters(), lr=lr, eps=eps)
    
    total_steps = len(train_dl)*epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    torch.cuda.empty_cache()
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}')
        model_bert.train()
        loss_train_total = 0
        
        b_count = 1
        for batch in train_dl:
            print(f'Epoch {epoch+1} - Batch {b_count}')
            #print(f'CUDA memory allocated: {torch.cuda.memory_allocated(0)}')
            b_count += 1
            model_bert.zero_grad()
            
            batch = tuple(b.to(device) for b in batch)
            
            outputs = model_bert(input_ids=batch[0],
                                 attention_mask = batch[1],
                                 labels = batch[2])
            loss = outputs[0]
            #logits = outputs[1] 
            
            loss_train_total += loss.item()
            # By default, pytorch expects backward() to be called for the last output of the network - the loss function. 
            # The loss function always outputs a scalar and therefore, 
            # the gradients of the scalar loss w.r.t all other variables/parameters is well defined (using the chain rule).
            loss.backward()
            
            optimizer.step()
            scheduler.step()
            
        loss_train_avg = loss_train_total/len(train_dl)
        loss_valid_avg, y_pred, y_true = eval_BERT(model_bert,test_dl,device)
        f1 = f1_score_BERT(y_pred, y_true)
        print(f"Training loss: {loss_train_avg}")
        print(f"Validation loss: {loss_valid_avg}")
        print(f"f1 score: {f1}")
    
    return model_bert













