import pickle
from FF_EDFST_Training.utils import model_utils2

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
    filepath = 'FF_EDFST_Training/exported_models/scaler.pkl'
    with open(filepath,'rb') as f:
        scaler_data = pickle.load(f)
    model_path='FF_EDFST_Training/exported_models/conv_model.tflite'
    
    data = data.tail(24)
    print(len(data))
    dataTestdf = data[['w_t300_current_level', 'w_t301_current_level', 'w_t302_current_level', 'w_t303_current_level','remaining_total_water_capacity','number_of_inlets']]
    data, mape,rmse, count = model_utils2.run_inference(model_path,dataTestdf,scaler_data)

    return data


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
