from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Dense, Input, Dropout
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam

# maxnorm helps limit the max size of a weight
from keras.constraints import maxnorm


def cl_lstm(x_train, y_train, x_test, y_test, word_embedding_layer):
    epoch = 10
    batch = 1024
    # used in the optimizer
    # 0.05 is outperformed by 0.01
    # always look into making it smaller when model isnot performing well
    alpha = 0.01
    dropout_ratio = 0.5
    # dimension of output space: # nodes in the dense layer
    # 200 performs way better than 100 in the first few epochs, 
    # but in the end, there is no discerable difference between the two
    LSTM_unit = 200
    LSTM_maxnorm = 4
    
    #validation set in training data
    val_split = 0.1
    
    model = Sequential()
    model.add(word_embedding_layer)
    model.add(Dropout(dropout_ratio))
    model.add(Bidirectional(LSTM(LSTM_unit,
                                 activation='tanh',
                                 recurrent_activation='sigmoid',
                                 kernel_constraint=maxnorm(LSTM_maxnorm),
    #                             dropout=0.2,
    #                             recurrent_dropout=0.2
                                 )))
    model.add(Dense(1, activation='sigmoid'))
    
    model.summary()
    
    adam = Adam(learning_rate=alpha)
    
    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    # other metrics to try: binary_accuracy,precision, recall, AUC
    
    ear_stop = EarlyStopping(patience=3, verbose=1, restore_best_weights=True)
    red_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=0.001, verbose=1)
    callbacks = [ear_stop,red_lr]
    
    # each batch trains on len(x_train)/batch*(1-valid_split)
    history = model.fit(
        x_train,
        y_train,
        batch_size = batch,
        epochs = epoch,
        validation_split=val_split,
        verbose=1,
        callbacks = callbacks
        )
    return model



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    