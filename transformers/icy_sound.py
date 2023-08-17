from FF_EDFST_Training.utils import tf_ff_edfst_models, model_utils2
import pandas as pd

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdatespip 
import seaborn as sns
import warnings
import math
import time
import os
from sklearn import preprocessing
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(data, *args, **kwargs):
    """
    Template code for a transformer block.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    # Specify your transformation logic here
    df = data[0]
    w_columns = data[1]
    print(w_columns)
    print('here')
    df['remaining_total_water_capacity'] = 1224 - df['sum_water_current_level']
    columns =  ['remaining_total_water_capacity','number_of_inlets','day_rolling_sum', 'day_rolling_sum_o']
    columns.extend(w_columns)
    print(columns)
    
    
    dfModel = df[columns]

    date_time = pd.to_datetime(df.pop('date_time'))
    dfModel = dfModel.fillna(0.00)
    column_indices = {name: i for i, name in enumerate(dfModel.columns)}
    cut = 0.86

    n = len(dfModel)
    train_df = dfModel[:int(n*cut)]
    test_df = dfModel[int(cut*n):]

    num_features = dfModel.shape[1]


    scaler_data = StandardScaler()
    scaler_data.fit(train_df)
    train_df  =  pd.DataFrame(scaler_data.fit_transform(train_df),columns = train_df.columns)
    test_df = pd.DataFrame(scaler_data.fit_transform(test_df),columns=test_df.columns)  

    df_std = train_df.melt(var_name='Column', value_name='Normalized')
    plt.figure(figsize=(12, 6))
    ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
    _ = ax.set_xticklabels(dfModel.keys(), rotation=90)


        
    multi_val_performance = {}
    multi_performance = {}


    OUT_STEPS = 18
    multi_window = tf_ff_edfst_models.WindowGenerator(input_width=24,train_df=train_df,val_df=test_df,test_df=test_df,
                                label_width=OUT_STEPS,
                                shift=OUT_STEPS)

    multi_window.plot(plot_col='remaining_total_water_capacity')


    multi_conv_model = model_utils2.create_cnn_lstm_skip_model(18,8)
    model_utils2.compile_and_fit(multi_conv_model,multi_window)
    





    ts = int(time.time())
    file_path = f"FF_EDFST_Training/exported_models/{ts}/"
    multi_conv_model.save(filepath=file_path, save_format='tf')
    converter = tf.lite.TFLiteConverter.from_keras_model(multi_conv_model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter._experimental_lower_tensor_list_ops = False
    tflite_model = converter.convert()
    with open(os.path.join('FF_EDFST_Training/exported_models/','conv_model.tflite'), 'wb') as f:
        f.write(tflite_model)

    import pickle
    with open(os.path.join('FF_EDFST_Training/exported_models/','scaler.pkl'),'wb') as f:
        pickle.dump(scaler_data,f)

   #cv,mape,rmse,count =  tf_ff_edfst_models.testModel(multi_conv_model,test_df,data,cut,scaler_data)
    #fig = px.line( cv.reset_index(), x= 'index', y=["actual","predictions"],width=1800, height=600, title = str(count) +" Predictions,   MAPE :"+str(mape)+"  RMSE : "+str(rmse))
    #fig.show()

    return [test_df,df,cut]


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
