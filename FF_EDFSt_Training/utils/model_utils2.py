import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdatespip 
import seaborn as sns
import warnings
import math
import time

from sklearn import preprocessing
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

label_column = 'remaining_total_water_capacity'

OUT_STEPS = 18
num_features= 8

MAX_EPOCHS = 20


print('hello')

def testModel(model_path,test_df,dfFilter,cut,scaler_data):
    print(model_path)
    count=0 
    appended_results=[]
    rmse=[]
    start=0
    step_size=18
    window_size=24
    features = num_features
    print('in testModel')

    interpreter = tf.lite.Interpreter(model_path)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.allocate_tensors()
    
    for i in range(0,len(test_df)):
        if(window_size+start+step_size>len(test_df)):
            break
        #make prediction 

        interpreter.set_tensor(input_details[0]['index'], test_df[start:start+window_size].to_numpy(dtype='float32').reshape(1,window_size,features))
        interpreter.invoke()
        ouh = interpreter.get_tensor(output_details[0]['index'])    
        tempdf = pd.DataFrame(scaler_data.inverse_transform(test_df[window_size+start:window_size+start+step_size])[:,0],columns=['actual'])
        
        #scale the predictions back to normal values
        tempdf['predictions'] = pd.Series(scaler_data.inverse_transform(np.squeeze(ouh))[:,0])
        tempdf.reset_index()
        tempdf.plot.line()
        n = np.sqrt(mean_squared_error(tempdf.actual.values,tempdf.predictions.values))
        print(n)
        rmse.append(n)
        count=count+1
        tempdf['peak'] = dfFilter[int(n*cut)+window_size+start:int(n*cut)+window_size+start+step_size].reset_index()[['peak_hours']]


        start=start+step_size
        appended_results.append(tempdf)
    appended_results = pd.concat(appended_results)


    print(count, 'Predictions')  
    rmse = np.sqrt(mean_squared_error(appended_results.actual.values,appended_results.predictions.values))
    mape = mean_absolute_percentage_error(appended_results.actual.values,appended_results.predictions.values)*100
    print('RMSE : ',rmse)
    print('MAPE',mape)
    
    capacity_constrained = appended_results[appended_results['actual']<400].reset_index()
    print(len(capacity_constrained), '< 400m^3 data points')  
    print('< 400m^3 RMSE : ',np.sqrt(mean_squared_error(capacity_constrained.actual.values,capacity_constrained.predictions.values)))
    print('< 400m^3 MAPE : ',mean_absolute_percentage_error(capacity_constrained.actual.values,capacity_constrained.predictions.values)*100)    
    peak_hours = appended_results[appended_results['peak']==1].reset_index()
    print(len(peak_hours), ': peak hour data points')  
    print('Peak hour RMSE : ',np.sqrt(mean_squared_error(peak_hours.actual.values,peak_hours.predictions.values)))
    print('Peak hour MAPE : ',mean_absolute_percentage_error(peak_hours.actual.values,peak_hours.predictions.values)*100)
    
    non_peak_hours = appended_results[appended_results['peak']==0].reset_index()
    print(len(non_peak_hours), ': non-peak hour data points')  
    print('Non - Peak hour RMSE : ',np.sqrt(mean_squared_error(non_peak_hours.actual.values,non_peak_hours.predictions.values)))
    print('Non - Peak hour MAPE : ',mean_absolute_percentage_error(non_peak_hours.actual.values,non_peak_hours.predictions.values)*100)    
    return appended_results.reset_index().drop(columns='index'), round(mape,2), round(rmse,2), count



def run_inference(model_path,test_df,scaler_data):
    print(model_path)
    count=0 
    appended_results=[]
    rmse=[]
    start=0
    step_size=18
    window_size=24
    features = num_features
    print('in testModel')

    interpreter = tf.lite.Interpreter(model_path)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.allocate_tensors()

    test_df[test_df.columns] = scaler_data.fit_transform(test_df[test_df.columns])
    interpreter.set_tensor(input_details[0]['index'], test_df.to_numpy(dtype='float32').reshape(1,window_size,features))
    interpreter.invoke()
    ouh = interpreter.get_tensor(output_details[0]['index']) 

    #tempdf = pd.DataFrame(scaler_data.inverse_transform(test_df[window_size+start:window_size+start+step_size])[:,0],columns=['actual'])
    
    #scale the predictions back to normal values

    
    return (pd.Series(scaler_data.inverse_transform(np.squeeze(ouh))[:,0]).to_json())




def compile_and_fit(model, window, patience=4):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

  model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=[tf.keras.metrics.MeanAbsoluteError()])

  history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping])
  return history




def create_simple_conv_model():
        CONV_WIDTH = 9
        return tf.keras.Sequential([
        # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
        tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
        # Shape => [batch, 1, conv_units]
        tf.keras.layers.Conv1D(120, activation='relu', kernel_size=(CONV_WIDTH),strides=1),
        
        # Shape => [batch, 1,  out_steps*features]
        tf.keras.layers.Dense(OUT_STEPS*num_features,
                            kernel_initializer=tf.initializers.zeros()),
        # Shape => [batch, out_steps, features]
        tf.keras.layers.Reshape([OUT_STEPS, num_features])
    ])


def create_lstm_csnn_model():
        CONV_WIDTH = 18
        return tf.keras.models.Sequential([
        tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
    # Shape => [batch, 1, conv_units]
        tf.keras.layers.Conv1D(120, activation='relu', kernel_size=(CONV_WIDTH),strides=2),
        tf.keras.layers.Conv1D(60, activation='relu',kernel_size=1,strides=2),
       
        tf.keras.layers.LSTM(32, activation='relu', return_sequences=True),
        tf.keras.layers.LSTM(16, activation='relu', return_sequences=False),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024),
        tf.keras.layers.Dense(OUT_STEPS*num_features, kernel_initializer=tf.initializers.random_uniform()),
        tf.keras.layers.Reshape([OUT_STEPS, num_features])
    ], name="lstm_cnn")


def create_cnn_lstm_skip_model(OUT_STEPS,num_features):
    tf.keras.backend.clear_session()
    inputs = tf.keras.layers.Input(shape=(24, num_features), name='main')
    sliced_inputs = tf.keras.layers.Lambda(lambda x: x[:, -OUT_STEPS:, :])(inputs)
    #inputs = tf.keras.layers.Input(shape=(n_steps,n_features), name='main')(prep)
    
    conv1 = tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu',strides=1)(sliced_inputs)
    max_pool_1 = tf.keras.layers.MaxPooling1D(2)(conv1)
    conv2 = tf.keras.layers.Conv1D(32, kernel_size=2, activation='relu')(max_pool_1)
    max_pool_2 = tf.keras.layers.MaxPooling1D(2)(conv2)
    lstm_1 = tf.keras.layers.LSTM(32, activation='relu', return_sequences=True)(max_pool_2)
    lstm_2 = tf.keras.layers.LSTM(16, activation='relu', return_sequences=False)(lstm_1)
    flatten = tf.keras.layers.Flatten()(lstm_2)
    
    skip_flatten = tf.keras.layers.Flatten()(inputs)

    concat = tf.keras.layers.Concatenate(axis=-1)([flatten, skip_flatten])
    dense_1 = tf.keras.layers.Dense(128, activation='relu')(concat)
    dense_2 = tf.keras.layers.Dense(OUT_STEPS*num_features,kernel_initializer=tf.initializers.random_uniform())(dense_1)
    out = tf.keras.layers.Reshape([OUT_STEPS,num_features])(dense_2)

    model = tf.keras.Model(inputs=inputs, outputs=out, name='lstm_skip')
    
    loss = tf.keras.losses.Huber()
    optimizer = tf.keras.optimizers.Adam()
    
    model.compile(loss=loss, optimizer='adam', metrics=['mae'])
    
    return model








  

