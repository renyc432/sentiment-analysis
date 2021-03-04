from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import spacy
import re

# preprocess data
# 1. lower case
# 2. replace special char
# 3. remove hyperlinks
# 4. remove @ and the following word
# 5. stemming the word
# 6. remove extra whitespace (taken care of in .split())

# nltk approach
stop = stopwords.words('english')
snowball = SnowballStemmer('english')
def text_preprocess_nltk(text, regex):
    # strip removes whitespace at beginning and end
    text = re.sub(regex, ' ', text.lower(), flags=re.I).strip()
    # by not specifying ' ', split() treats multiple consecutive whitespaces as one
    word_token = text.split()
    token_clean = [snowball.stem(w) for w in word_token if w not in stop]
    return ' '.join(token_clean)


# spacy approach
nlp = spacy.load('en_core_web_sm', disable=['ner','parser'])
def text_preprocess_spacy(text, regex):
    text = re.sub(regex, ' ', text.lower(), flags=re.I).strip()
    
    token_clean = [w.lemma_ for w in nlp(text) if not w.is_stop]
    return ' '.join(token_clean)



def text_preprocess(text, text_regex, mode='nltk'):
    if mode=='nltk':
        return text_preprocess_nltk(text,text_regex)
    else:
        return text_preprocess_spacy(text, text_regex)
    