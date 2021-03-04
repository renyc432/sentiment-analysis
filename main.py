import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

############################### Hyperparameters ################################
path = 'C:\\Users\\roy79\\Desktop\\Research\\sentiment-analysis'
data_encoding = 'ISO-8859-1'

text_regex = "http\S+|@\S+|[^a-zA-Z0-9]+"

method = 'LSTM'

test_split_ratio = 0.2
test_split_seed = 123
valid_seed = 441

glove_path = 'glove.6B.300d.txt'
pad_maxlen = 100

# my laptop can only handle 16; 32 is too large, doesn't fit in GPU memory; try reducing sequence size to fit a 32 batch_size
batch_size_bert = 16
epochs_bert = 4

# adamw parameters
lr_adamw = 2e-5
eps_adamw = 1e-6


os.chdir(path)
################################ Import Data ###################################
from text_preprocess import text_preprocess
from embedding import glove
from cl_LSTM import cl_lstm
from evaluate import model_eval
from evaluate import predict_sentiment


dataset = pd.read_csv('twitter_1.6M.csv', encoding=data_encoding, header=None, 
                      names=['sent','id','date','flag','user','text'])

dataset.head()
twitter = dataset[['sent','text']]

#dataset = pd.read_csv('https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv')
#dataset.head()
#twitter = dataset[['label','tweet']]
#twitter = twitter[0:10000]
#twitter.columns = ['sent','text']
twitter.head()


############################ Exploratory Analysis ##############################
num_null_text = sum(twitter.text.isnull())
num_null_sent = sum(twitter.sent.isnull())
print(f'number of null text: {num_null_text}')
print(f'number of null sentiment: {num_null_sent}')

# check for imbalanced classes
count_emo = twitter['sent'].value_counts()
fig, ax = plt.subplots(figsize = (12,10))
sns.barplot(x=['negative','positive'],y=count_emo)

# select random text to see the kind of text we are working with
text_sample = twitter.iloc[random.choices(range(0, len(twitter)),k=20),1]
pd.options.display.max_colwidth = 100
print(text_sample)
# messy data: 
# repeated punctuations (... or !!!)
# remove @ and the word that follows
# remove hyperlinks
# hashtags, should provide some useful information
# emoji: how to remove them?


if method == 'LSTM':
    twitter['text'] = [text_preprocess(sentence,text_regex) for sentence in twitter['text']]
    #twitter['text'] = [text_preprocess_spacy(sentence) for sentence in twitter['text']]
    text_sample = twitter.iloc[random.choices(range(0,len(twitter)),k=20),1]
    print(text_sample)
    
    train, test = train_test_split(twitter, 
                                   test_size=test_split_ratio, 
                                   random_state = test_split_seed)
    #train.head()
    print(f'Size of training: {len(train)}')
    print(f'Size of test: {len(test)}')
    
    # tokenizer chops up sentences into words and map them to an index using a dictionary
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train.text)
    word_index = tokenizer.word_index
    # row dim (# of rows) of the word embedding layer matrix
    vocab_size = len(word_index) + 1
    print(vocab_size)
    
    # texts_to_sequences: only words known by the tokenizer will be taken into account
    x_train = pad_sequences(tokenizer.texts_to_sequences(train.text),
                            maxlen = pad_maxlen)
    x_test = pad_sequences(tokenizer.texts_to_sequences(test.text),
                           maxlen = pad_maxlen)
    
    ##################################### LSTM #####################################
    def y_encoding(train, test):
        # this maps the sentiment value to be binary (0,1)
        # not sure if this step is necessary, since the target is already binary
        encoder = LabelEncoder()
        # find list of unique values in y ([0,4] in this case)
        encoder.fit(twitter.sent.to_list())
        print('list of class labels:', encoder.classes_)
        y_train = encoder.transform(train.sent.to_list())
        y_test = encoder.transform(test.sent.to_list())
        
        # reshape turns the array into a matrix form where one of the dimension can be -1
        # this means numpy will figure out what that dimension is based on the other dimension provided
        y_train = y_train.reshape(-1,1)
        y_test = y_test.reshape(-1,1)
        return [y_train, y_test]
    
    y = y_encoding(train,test)
    y_train = y[0]
    y_test = y[1]
    
    glove_path = 'glove.6B.300d.txt'
    word_embedding_layer = glove(glove_path, vocab_size, word_index, pad_maxlen)
    
    
    if tf.config.list_physical_devices('GPU'):
        print('GPU is available')
    else:
        print('GPU is not available')
    
    model_lstm = cl_lstm(x_train, y_train, x_test, y_test, word_embedding_layer)
    model_eval(model_lstm, x_test, y_test)
    
    model_lstm.save('bidirectional LSTM model.h5')
    
elif (method == 'BERT'):
    import cl_BERT
    
    pretrained = 'bert-base-uncased'
    
    dataset_bert = cl_BERT.tokenize(data=twitter, col_x='text', col_y='sent', 
                                    pad_maxlen = pad_maxlen,
                                    pretrained_model=pretrained)
    
    train,test = cl_BERT.load(dataset_bert, 
                              test_ratio = test_split_ratio, 
                              test_seed = test_split_seed, 
                              batch_size=batch_size_bert)
    
    model_bert = cl_BERT.finetune(train, test, epochs_bert, lr_adamw, eps_adamw, pretrained, 
                                  valid_seed=valid_seed)
    
    model_bert.save('bert model.h5')



string = 'I want to fly to the moon'
string = 'I hate that'
predict_sentiment(model_lstm, tokenizer,string, pad_maxlen)











