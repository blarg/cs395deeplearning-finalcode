import scipy.io as spio
import numpy as np
import math
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from collections import Counter


def readMat(dataPath):
    
    
    #readMat: Reads the .mat for mad river data files placed at data path
    #INPUT
    #dataPath == relative location of the data path to the code file, or absolute
    #
    #OUTPUT
    #still being worked on
    #eventToClass== np array of all classes belonging to all 616events. value at 0 index is class for 0th event.
    #sedimentData_events == list containing array for sediment data for each event
    #
    #
    #events == list containing 2-D array for each event. 1st column is sediment and 2nd is stream flow. Ideally we will be using this
    #maxEventLen == Longest event in terms of timesteps
    
    #sample call: eventToClass, myEvents, maxEventLen,streamFlow_Data,sedimentData_events = readMat('..\data\')
    
    #Programmer: Ali Javed
    #Date last modified: 27 Feb 2018
    #modified by: Ali Javed
    #Comments: Initial version.
    
    
    ##############################################################################################################
        
    
    
    
    classMat = spio.loadmat(dataPath + 'allMadSitesStormHystClassKey.mat', squeeze_me=True)
    dataMat = spio.loadmat(dataPath + 'allMadSitesEventTimeSeries.mat', squeeze_me=True)
    
    eventToClass = classMat['stormHystClass'][:,3] #index 3 refers to class of 3rd event. Event number start from 0
    eventToClass = eventToClass.astype(int) # we do not need float classes




    #gather 626 events
    events = []
    sedimentData_events = []
    streamFlowData_events = []
    counter = 0
    maxEventLen = -1 #need this as fixed size input to keras RNN
    
    streamFlow = 1
    suspendedSedimentConcentration = 2
    
    for event in range(0,len(dataMat['dataTSOut'])):
    
    
        #not reading datetime and rainfall data for now
        #event_dataTime = np.zeros((len(dataMat['dataTSOut'][event][streamFlow]))) #can not extract datetime so setting it to one, for out purpose it does not matter anyways
        #event_rainFall = np.zeros((len(dataMat['dataTSOut'][event][streamFlow])))
                             
        event_streamflow = dataMat['dataTSOut'][event][streamFlow]
        event_suspendedSedimentConcentration = dataMat['dataTSOut'][event][suspendedSedimentConcentration]
    
        
        eventArray = np.column_stack((event_streamflow,event_suspendedSedimentConcentration))
        
        
        events.append(eventArray)
        sedimentData_events.append(event_streamflow)
        streamFlowData_events.append(event_suspendedSedimentConcentration)
    
        if len(event_streamflow)> maxEventLen:
            maxEventLen = len(event_streamflow)
    
    
        
        ##############################################################################################################
        #for classification based only on rain and sediment... i can not figure out how to give 2d input to RNN
        
        
        #classVector = np.repeat(eventToClass[event], len(event_streamflow))
        #print(np.shape(classVector))
        #print(np.shape(suspendedSedimentConcentration))
        #streamFlow_Data = np.column_stack((event_streamflow,classVector))
        #suspendedSedimentConcentration_Data = np.column_stack((event_suspendedSedimentConcentration,classVector))
        
    return eventToClass, events, maxEventLen, streamFlowData_events, sedimentData_events
    
    
 ##############################################################################################################
       


def readCharMat(dataPath):


    dataMat = spio.loadmat(dataPath + 'mixoutALL_shifted.mat', squeeze_me=True)
    classMat = spio.loadmat(dataPath +'char_class_labels',squeeze_me=True)  
    eventToClass = classMat['y_data']

    #gather 626 events
    events = []
    sedimentData_events = []
    streamFlowData_events = []

    maxEventLen = -1 #need this as fixed size input to keras RNN

    streamFlow = 1
    suspendedSedimentConcentration = 2

    for event in range(0,len(dataMat['mixout'])):


            event_streamflow = dataMat['mixout'][event][streamFlow]
            event_suspendedSedimentConcentration = dataMat['mixout'][event][suspendedSedimentConcentration]


            eventArray = np.column_stack((event_streamflow,event_suspendedSedimentConcentration))


            events.append(eventArray)
            sedimentData_events.append(event_streamflow)
            streamFlowData_events.append(event_suspendedSedimentConcentration)

            if len(event_streamflow)> maxEventLen:
                maxEventLen = len(event_streamflow)


    return eventToClass, events, maxEventLen, streamFlowData_events, sedimentData_events
 
    


 



 
    


dataPath = '../data/'

#eventToClass as an array of len(events) with each index telling the class of event
#myEvents is a list containing 2-d arrays for all events. 0 column is the stream flow, 1 column is sediment concentration
#eventToClass, myEvents, maxEventLen,streamFlow_Data,sedimentData_events = readMat(dataPath)
eventToClass, myEvents, maxEventLen,streamFlow_Data,sedimentData_events = readCharMat(dataPath)


import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import RNN
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.layers import Activation
from keras.layers import Bidirectional
# fix random seed for reproducibility
numpy.random.seed(7)




#append two arrays into single for each even, apparantely it doesnt matter what order data is passed to NN for 2 d events, as long as format is consitent
#i.e first 20 rows for sediment rate, next 20 for rain
##################################################
#one inputs at every time step
'''
rain_sediment = []
for i in range(0,len(sedimentData_events)):
    t1 = sedimentData_events[i]
    t2 = streamFlow_Data[i]
    
    rain_sediment_event = np.append(t1,t2)
    rain_sediment.append(rain_sediment_event)
'''

##################################################
#two inputs at every time step
rain_sediment = []
for i in range(0,len(sedimentData_events)):    
    t1 = sedimentData_events[i]
    t2 = streamFlow_Data[i]
    
    rain_sediment_event = np.column_stack((t1,t2))
    rain_sediment.append(rain_sediment_event)
    
#################
#this lenght 313 is the lenght of max stream flow event, hence if need feature extraction, use this

#############
maxEventLen = 313

#preprocess data to required format, padding to make all sequence data same lenght
#to use sediment rate onle, replace with sedimentData_events, for stream for data only streamFlow_Data, for both sediment and stream flow rain_sediment 
X_data = sequence.pad_sequences(rain_sediment,dtype='float',maxlen = maxEventLen)
#create one hot representation for class values
y_dataOneHot = to_categorical(eventToClass, num_classes=None)
y_dataCont = eventToClass


#Test/Train Split

sampler = StratifiedKFold(n_splits=4, shuffle=False, random_state=None)
#sampler.split(x= X_data,y=eventToClass)

for train_index, test_index in sampler.split(X_data, eventToClass):
    X_train, X_test = X_data[train_index], X_data[test_index]
    y_trainOneHot, y_testOneHot = y_dataOneHot[train_index], y_dataOneHot[test_index]
    y_trainCont, y_testCont = y_dataCont[train_index], y_dataCont[test_index]
    
    
    
#############################################################################
#DECLARE PARAMETERS FOR NN
#these parameters are the architecture of the RNN. Still have to do more on this
embedding_vecor_length = 16  #each event is represented using a 32 length vector. 
epochs = 700
batchSize = len(X_train)  #use all data.
maxEventLenPram = maxEventLen
#maxEventLenPram = maxEventLen *2 # if we are passing both rain and sediment in 1 d use multiply maxevent lenght by 2, this is input size if trying only rain or sediment, use maxeventlen
m = np.amax(X_train)+1 #what is the maximum data value
m = round(m)
len_input = int(m)
num_classes = y_testCont[1]+1
max_value = int(np.amax(X_train)+1)
###############################################################################



X_train = X_train.reshape(len(X_train),maxEventLenPram , 2)
X_test = X_test.reshape(len(X_test), maxEventLenPram, 2)



###############################################################################
#CREATE SIMPLE RNN
from keras.layers import Input

# create the model

model = Sequential()


cells = [
    LSTM(100),
    LSTM(100),
    LSTM(100),
]
#model.add(LSTM(128, input_shape=(maxlen, len(chars))))

#model.add(Embedding(input_dim = max_value, output_dim = maxEventLenPram, input_length=maxEventLenPram))
#add the recurrent LSTM layer of 100 nodes
#model.add(RNN(cells))
model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=(maxEventLenPram, 2)))
model.add(Bidirectional(LSTM(128)))




#model.add(TimeDistributed(Dense(1, activation='sigmoid')))
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(len(y_trainOneHot[0])))
model.add(Activation('softmax'))
#model.add(LSTM(100))
#out put layer with 16 nodes, one hot representation

#model.add(TimeDistributed(Dense(num_classes+1), input_shape=(maxEventLenPram, 1)))
#model.add(Dense(num_classes+1, activation='relu'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())





import os.path
#if os.path.exists('../history/3Layer1000Weights'):
    #model.load_weights('../history/3Layer1000Weights')
    #print('Weights loaded')
    
    
history = model.fit(X_train, y_trainOneHot, epochs=epochs, batch_size=batchSize)


#model.save_weights('../history/biDir_3layer_char_50205.weights')
model.save_weights('../history/biDir_LSTM128128_D6464.weights')

   
import pickle        
path = '../history/biDir_LSTM128128_D6464.his'                                          
with open(path, 'wb') as file_pi:
    pickle.dump(history.history, file_pi)


loss, acc = model.evaluate(X_test, y_testOneHot)
print("Model accuracy on test ",acc)
print("Model loss on test ",loss)



