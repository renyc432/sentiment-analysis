import re
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import string
import nltk
import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)

#%matplotlib inline



#path = 'C:\\Users\\rs\\Desktop\\Datasets\\NLP\\twitter_1.6M.csv'


train = pd.read_csv('https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv')
train_original=train.copy()

test = pd.read_csv('https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/test.csv')
test_original=test.copy()


positive = 0
negative = 1



combine = train.append(test,ignore_index=True, sort=True)


def remove_pattern(text, pattern):
    r = re.findall(pattern, text)
    
    for i in r:
        # sub all occurence of i in text for ''
        text = re.sub(i,'',text)
        
    return text


combine['Tidy_Tweets'] = np.vectorize(remove_pattern)(combine['tweet'],'@[\w]*')
combine.head()

combine['Tidy_Tweets'] = combine['Tidy_Tweets'].str.replace('^a-zA-Z#',' ')