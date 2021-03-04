from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score

from keras.preprocessing.sequence import pad_sequences

import torch
import numpy as np

def model_eval(model, x_test, y_test):
    '''Output: accuracy, recall, precision, f1, auc_roc'''
#    score = model.evaluate(x_test,y_test,batch_size=batch)
#    print('loss:', score[0])
#    print('accuracy:', score[1])
    
    scores = model.predict(x_test, verbose=1)    
    y_pred = [1 if score > 0.5 else 0 for score in scores]
    acc_lstm = accuracy_score(y_test, y_pred)
    f1_lstm = f1_score(y_test, y_pred)
    rec_lstm = recall_score(y_test,y_pred)
    prec_lstm = precision_score(y_test, y_pred)
    auc_lstm = roc_auc_score(y_test, y_pred)

    print(f'LSTM - Accuracy: {acc_lstm}')
    print(f'LSTM - recall: {rec_lstm}')
    print(f'LSTM - precision: {prec_lstm}')
    print(f'LSTM - f1: {f1_lstm}')
    print(f'LSTM - auc_roc: {auc_lstm}')
    
    
def predict_sentiment(model, tokenizer, string, maxlen):
    
    seq = pad_sequences(tokenizer.texts_to_sequences([string]), maxlen = maxlen)
    prob = model.predict(seq)[0][0]
    
    return 'Positive' if prob>0.5 else 'Negative'



def eval_BERT(model, test_dl, device):
    model.eval()
    
    loss_valid_total = 0
    y_pred = []
    y_true = []
    
    
    for batch in test_dl:
        batch = tuple(b.to(device) for b in batch)
        
        with torch.no_grad():
            outputs = model(input_ids = batch[0],
                            attention_mask = batch[1],
                            labels = batch[2])
        
        loss_valid_total += outputs[0].item()
        y_pred.append(outputs[1].detach().cpu().numpy())
        y_true.append(batch[2].detach().cpu().numpy())
    
    loss_valid_avg = loss_valid_total/len(test_dl)
    y_pred = np.concatenate(y_pred, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    return loss_valid_avg,y_pred,y_true


# for BERT
def f1_score_BERT(y_pred, y_test):
    y_pred_flat = np.argmax(y_pred,axis=1).flatten()
    # all are 0
    y_test_flat = y_test.flatten()
    return f1_score(y_test_flat, y_pred_flat)





