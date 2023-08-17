if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test
import pickle
from FF_EDFST_Training.utils import model_utils2
@data_loader
def load_data(*args, **kwargs):
    """
    Template code for loading data from any source.

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    # Specify your data loading logic here

    filepath = 'FF_EDFST_Training/exported_models/scaler.pkl'
    with open(filepath,'rb') as f:
        scaler_data = pickle.load(f)
    model_path='ff_edfst/exported_models/conv_model.tflite'
# Load TFLite model and allocate tensors.
    
    # Get input and output tensors.
    #print(interpreter,data[0],data[1],data[2])
    data, mape,rmse, count = model_utils2.testModel(model_path,data[0],data[1],data[2],scaler_data)


    return data


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
