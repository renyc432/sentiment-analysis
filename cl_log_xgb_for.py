import re
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import nltk
from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image
import requests
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
#%matplotlib inline



#path = 'C:\\Users\\rs\\Desktop\\Datasets\\NLP\\twitter_1.6M.csv'


train = pd.read_csv('https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv')
train_original=train.copy()

test = pd.read_csv('https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/test.csv')
test_original=test.copy()

#positive = 0
#negative = 1

# Combine train and test because tweets in test also contain information
combine = train.append(test,ignore_index=True, sort=True)


def remove_pattern(text, pattern):
    r = re.findall(pattern, text)
    
    for i in r:
        # sub all occurence of i in text for ''
        text = re.sub(i,'',text)
        
    return text

# remove tweeter handlers
# np.vectorize is faster than for loop
combine['Tidy_Tweets'] = np.vectorize(remove_pattern)(combine['tweet'],'@[\w]*')
combine.head()

# replace none a-z characters (numbers, special characters, punctuations) with ' '
combine['Tidy_Tweets'] = combine['Tidy_Tweets'].str.replace('^a-zA-Z#',' ')

# remove short words as they usually do not contain significant information (such as 'a', 'an', 'hmm')
# This removes any word 3 characters or shorter
# This joins words with 4+ characters
combine['Tidy_Tweets'] = combine['Tidy_Tweets'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))


# Split each string into words
tokenized_tweet = combine['Tidy_Tweets'].apply(lambda x: x.split())
tokenized_tweet.head()


# Stemming strips the suffixes ('ing', 'ly', 's') from a word
# So 'player', 'played', 'plays', 'playing' all turn into play 

# Look up how this works
ps = nltk.PorterStemmer()
# This is costly
tokenized_tweet = tokenized_tweet.apply(lambda x: [ps.stem(i) for i in x])

tokenized_tweet.head()

# Put it back to a string
tokenized_tweet_temp = tokenized_tweet.apply(lambda x: ' '.join(x))

combine['Tidy_Tweets'] = tokenized_tweet_temp
combine.head()


# Store all words from dataset
all_words_positive = ' '.join(w for w in combine['Tidy_Tweets'][combine['label']==0])
all_words_negative = ' '.join(w for w in combine[combine['label']==1]['Tidy_Tweets'])


# Extract hashtags from tweets
def hashtags_extract(x):
    hashtags = []
    
    for i in x:
        # r: raw string, this resolves the problem that both python and regex uses \ for escape
        # \w: matches a-z/A-Z, 0-9,_
        # +: 1+ characters
        ht = re.findall(r'#(\w+)',i)
        hashtags.append(ht)
    
    return hashtags

ht_positive = hashtags_extract(combine['Tidy_Tweets'][combine['label']==0])
# Unnest nested list
ht_positive_unnest = sum(ht_positive,[])


ht_negative = hashtags_extract(combine['Tidy_Tweets'][combine['label']==1])
ht_negative_unnest = sum(ht_negative,[])





# bag-of-words
bag_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')

bag_matrix = bag_vectorizer.fit_transform(combine['Tidy_Tweets'])
bag_features = bag_vectorizer.get_feature_names()
df_bag = pd.DataFrame(bag_matrix.toarray(),columns=bag_features)

# TF-IDF


#max_df, min_df: ignore word if frequency/count pass the max/min
tfidf_vectorizer = TfidfVectorizer(max_df=0.9, min_df=2, max_features=1000,stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(combine['Tidy_Tweets'])
tfidf_features = tfidf_vectorizer.get_feature_names()
df_tfidf = pd.DataFrame(tfidf_matrix.toarray(),columns=tfidf_features)



train_bag = bag_matrix[:len(train)]
train_tfidf = tfidf_matrix[:len(train)]

# Train, validation split
x_train_bag,x_valid_bag,y_train_bag,y_valid_bag = train_test_split(train_bag,train['label'],
                                                                   test_size = 0.3, random_state = 2)
x_train_tfidf,x_valid_tfidf,y_train_tfidf,y_valid_tfidf = train_test_split(train_tfidf,train['label'],
                                                                           test_size = 0.3, random_state = 17)



# Logistic Regression
#L1 penalty: lasso; L2 penalty: ridge
Log_bag = LogisticRegression(random_state=0,solver='lbfgs')
Log_bag.fit(x_train_bag,y_train_bag)

log_pred_bag = Log_bag.predict_proba(x_valid_bag)
# why 0.3??
# first column = 0 (positive); second column = 1 (negative)
log_pred_int_bag = log_pred_bag[:,1]>=0.3
log_pred_int_bag = log_pred_int_bag.astype(np.int)
# calculate f1 score
f1_log_bag = f1_score(y_valid_bag, log_pred_int_bag)
auc_log_bag = roc_auc_score(y_valid_bag, log_pred_int_bag)

Log_tfidf = LogisticRegression(random_state=0,solver='lbfgs')
Log_tfidf.fit(x_train_tfidf, y_train_tfidf)
log_pred_tfidf = Log_tfidf.predict_proba(x_valid_tfidf)
log_pred_int_tfidf = log_pred_tfidf[:,1]>=0.3
log_pred_int_tfidf = log_pred_int_tfidf.astype(np.int)
f1_log_tfidf = f1_score(y_valid_tfidf, log_pred_int_tfidf)
auc_log_tfidf = roc_auc_score(y_valid_tfidf, log_pred_int_tfidf)


# XGBoost
xgb_bag = XGBClassifier(random_state=22,learning_rate=0.1, n_estimators=1000)
xgb_bag.fit(x_train_bag,y_train_bag)
xgb_pred_bag = xgb_bag.predict_proba(x_valid_bag)
xgb_pred_int_bag = (xgb_pred_bag[:,1]>= 0.3).astype(np.int)
f1_xgb_bag = f1_score(y_valid_bag, xgb_pred_int_bag)
auc_xgb_bag = roc_auc_score(y_valid_bag, xgb_pred_int_bag)
recall_xgb_bag = recall_score(y_valid_bag, xgb_pred_int_bag)
precision_xgb_bag = precision_score(y_valid_bag, xgb_pred_int_bag)
acc_xgb_bag = accuracy_score(y_valid_bag, xgb_pred_int_bag)

xgb_tfidf = XGBClassifier(random_state=22,learning_rate=0.1,n_estimators=1000)
xgb_tfidf.fit(x_train_tfidf,y_train_tfidf)
xgb_pred_tfidf = xgb_tfidf.predict_proba(x_valid_tfidf)
xgb_pred_int_tfidf = (xgb_pred_tfidf[:,1]>=0.3).astype(np.int)
f1_xgb_tfidf = f1_score(y_valid_tfidf, xgb_pred_int_tfidf)
auc_xgb_tfidf = roc_auc_score(y_valid_tfidf, xgb_pred_int_tfidf)
recall_xgb_tfidf = recall_score(y_valid_tfidf, xgb_pred_int_tfidf)
precision_xgb_tfidf = precision_score(y_valid_tfidf, xgb_pred_int_tfidf)
acc_xgb_tfidf = accuracy_score(y_valid_tfidf, xgb_pred_int_tfidf)


# Decision Trees
# criterion: gini or entropy (information gain)
tree_bag = DecisionTreeClassifier(random_state=1, criterion='entropy')
tree_bag.fit(x_train_bag,y_train_bag)
tree_pred_bag = tree_bag.predict_proba(x_valid_bag)
tree_pred_int_bag = (tree_pred_bag[:,1]>=0.3).astype(np.int)
f1_tree_bag = f1_score(y_valid_bag, tree_pred_int_bag)
auc_tree_bag = roc_auc_score(y_valid_bag, tree_pred_int_bag)


tree_tfidf = DecisionTreeClassifier(random_state=1,criterion='entropy')
tree_tfidf.fit(x_train_tfidf,y_train_tfidf)
tree_pred_tfidf = tree_tfidf.predict_proba(x_valid_tfidf)
tree_pred_int_tfidf = (tree_pred_tfidf[:,1]>=0.3).astype(np.int)
f1_tree_tfidf = f1_score(y_valid_tfidf, tree_pred_int_tfidf)
auc_tree_tfidf = roc_auc_score(y_valid_tfidf, tree_pred_int_tfidf)




# Word cloud
Mask = np.array(Image.open(requests.get('http://clipart-library.com/image_gallery2/Twitter-PNG-Image.png', stream=True).raw))
image_colors = ImageColorGenerator(Mask)
wc_0 = WordCloud(background_color='black', height = 1500, width = 4000, mask = Mask).generate(all_words_positive)

# Size of the image generated
plt.figure(figsize=(160,320))
plt.imshow(wc_0.recolor(color_func=image_colors),interpolation='hamming')

wc_1 = WordCloud(background_color='black', height = 1500, width = 4000, mask = Mask).generate(all_words_negative)
plt.figure(figsize=(160,320))
plt.imshow(wc_1.recolor(color_func=image_colors),interpolation='gaussian')


# Plot bar-plot
word_freq_positive = nltk.FreqDist(ht_positive_unnest) # This is similar to a dictionary
word_freq_negative = nltk.FreqDist(ht_negative_unnest) # This is similar to a dictionary

df_positive = pd.DataFrame({'Hashtags':list(word_freq_positive.keys()),'Count':list(word_freq_positive.values())})
df_positive_plot = df_positive.nlargest(20,columns='Count')
df_negative = pd.DataFrame({'Hashtags':list(word_freq_negative.keys()),'Count':list(word_freq_negative.values())})
df_negative_plot = df_negative.nlargest(20,columns='Count')

# seaborn library
sns.barplot(data=df_positive_plot,y='Hashtags',x='Count').set_title('Top 20 Positive Freq')
sns.despine()

sns.barplot(data=df_negative_plot,y='Hashtags',x='Count').set_title('TOP 20 Negative Freq')
sns.despine()