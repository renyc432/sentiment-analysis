import numpy as np
from keras.layers import Embedding

def glove(path, vocab_size, word_index, max_padding_length):
    # load pretrained word embedding layer
    glove_dict = {}
    
    with open(path,'r',encoding='utf-8') as f:
        for word in f:
            items = word.split()
            glove_dict[items[0]] = np.asarray(items[1:],dtype='float32')
    
    # This creates a matrix where each row represent a word in the document
    # embedding_dim is the right hand side dimension (number of neurons in the next layer)
    # it is determined randomly, or experiement with different numbers
    embedding_dim = 300
    embedding_mat = np.zeros((vocab_size, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = glove_dict.get(word)
        if embedding_vector is not None:
            embedding_mat[i] = embedding_vector
    
    # Embedding() is specifically for text data
    # input_dim=vocab_size: it's always the size of the vocabulary, ie. max integer index+1
    # output_dim: dimension of the dense embedding
    # input_length = pad_maxlen: because this is the length of input sequence
    #   experiment both with and without [], compare the result
    # why the [] with [embedding_mat]?
    # trainable=False: do not train this layer
    word_embedding_layer = Embedding(input_dim=vocab_size,
                                output_dim=embedding_dim,
                                weights=[embedding_mat],
                                input_length=max_padding_length,
                                trainable=False)
    
    return word_embedding_layer