from pandas import read_csv
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import numpy as np

dataset = read_csv('D:/some_data.txt', header=0, index_col=0, sep="\t")

def fetch_data(dataset,n_train_hours):
    values = dataset.values
    train = values[:n_train_hours, :]
    test = values[n_train_hours:, :]
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1] 
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    return train_X,train_y,test_X,test_y

def model_LSTM(train_X,activ,dropo):
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1],train_X.shape[2] ),dropout=dropo,return_sequences = True)) 
    model.add(LSTM(35,return_sequences = True))
    model.add(LSTM(10))
    model.add(Dense(1,activation=str(activ))) 
    return model

def save(model,method):
    if method =='json':
        modelJSON = model.to_json()
        with open('model.json','w') as json_file:
            json_file.write(modelJSON)
        model.save_weights('model.h5')
        return
    else:
        model.save('NBM.h5')

def estimate(test_X,train_X,yhat,xy,test_y,train_y):
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
    train_X = train_X.reshape((train_X.shape[0], train_X.shape[2]))
    inv_yhat = np.concatenate((yhat, test_X[:, 1:]), axis=1)
    inv_yhat = inv_yhat[:,0]
    
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_y, test_X[:, 1:]), axis=1)
    train_y = train_y.reshape((len(train_y), 1))
    invt_y = np.concatenate((train_y, train_X[:, 1:]), axis=1)
    
    inv_y = inv_y[:,0]
    invt_y = invt_y[:,0]
    
    xTrain=np.linspace(0,len(yhat),num=len(yhat))
    yhat= (yhat[:,0]*4.730348145)+62.4926759
    inv_y=(inv_y*4.730348145)+62.4926759
    
    xTest = np.linspace(0,len(xy),num=len(xy))
    xy=(xy[:,0]*4.730348145)+62.4926759
    invt_y=(invt_y*4.730348145)+62.4926759
    return yhat,inv_y,xy,invt_y,xTrain,xTest
                 
'''    
#COMMENT ONCE WEIGHTS SAVED
train_X,train_y,test_X,test_y=fetch_data(dataset,700)
model=model_LSTM(train_X,tanh,0.4)
model.compile(loss='mae', optimizer='adam')
model.fit(train_X, train_y, epochs=50, batch_size=100, validation_data=(test_X, test_y), verbose=2, shuffle=False)
#save(model,'')  
'''


''' LOAD JSON
json_file = open('model.json','r')
weights = json_file.read()
json_file.close()
load_model=model_from_json(weights)
load_model.compile(loss='mae', optimizer='adam')
'''



