import pandas as pd, re
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
    df  = pd.read_csv(data)

    df = df.drop(['master_date_time'], axis=1)
    # sort the dataframe by datetime values in ascending order
    df['master_date_time_mst'] = pd.to_datetime(df['master_date_time_mst'])
    df = df.sort_values('master_date_time_mst')
    df = df.rename(columns = {'master_date_time_mst': 'date_time'})
    weekdays = {0:'Monday', 1:'Tuesday', 2:'Wednesday', 3: 'Thursday', 4: 'Friday', 5:'Saturday', 6:'Sunday'}
    df['weekday'] = df.date_time.dt.weekday.map(weekdays)
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    elements = df.columns

    pattern = re.compile(r'w_t\d+_current_level')
    w_current_level_columns = list(filter(pattern.match, elements))
    print(w_current_level_columns)
    df['sum_water_current_level'] = df[w_current_level_columns].sum(axis=1)


    pattern = re.compile(r'o_t\d+_current_level')
    o_current_level_columns = list(filter(pattern.match, elements))
    print(o_current_level_columns)
    df['sum_oil_current_level'] = df[o_current_level_columns].sum(axis=1)


    pattern = re.compile(r'e_t\d+_current_level')
    e_current_level_columns = list(filter(pattern.match, elements))
    print(e_current_level_columns)
    df['sum_emulsion_current_level'] = df[e_current_level_columns].sum(axis=1)
    df['weekday'] = pd.Categorical(df['weekday'], ordered=True, categories=weekday_order)

    df['dayofweek'] =  df.date_time.dt.dayofweek


    peak_start = pd.to_datetime('07:00:00').time()
    peak_end = pd.to_datetime('19:00:00').time()

    # Create new column based on peak hours
    df['peak_hours'] = df['date_time'].apply(lambda x: 1 if peak_start <= x.time() <= peak_end else 0)

    pattern = re.compile(r'inlet[1-8]_flow_rate')
    
    cols_to_check = list(filter(pattern.match, elements))

# count the number of columns with values greater than 0 for each row
    df['number_of_inlets'] = df[cols_to_check].gt(0).sum(axis=1)

    df['Day'] = df.date_time.dt.date
    df = df.reset_index().drop(columns=['index'])
    df  = df.sort_values(by='date_time',ascending=True)

    df['day'] = df['date_time'].dt.floor('D')
    df['rolling_sum_o'] = df.set_index('day').groupby('day')['w_outlet_volume'].rolling('24h').sum().reset_index(drop=True)

    # Reset the rolling sum to zero at the start of each day, except for the first 24 hours of each day
    df['day_rolling_sum_o'] = df.groupby('day')['rolling_sum_o'].shift(1)
    df['day_rolling_sum_o'] = df['day_rolling_sum_o'].where(df['day'] == df['day'].shift(1), 0)
    df['rolling_sum_o'] = df['rolling_sum_o'] - df['day_rolling_sum_o']

    df['rolling_sum'] = df.set_index('day').groupby('day')['pure_water_total_volume'].rolling('24h').sum().reset_index(drop=True)

    # Reset the rolling sum to zero at the start of each day, except for the first 24 hours of each day
    df['day_rolling_sum'] = df.groupby('day')['rolling_sum'].shift(1)
    df['day_rolling_sum'] = df['day_rolling_sum'].where(df['day'] == df['day'].shift(1), 0)
    df['rolling_sum'] = df['rolling_sum'] - df['day_rolling_sum']

    df  = df[((df['date_time']>pd.to_datetime('2022-10-29 00:00:00')) & (df['date_time']<pd.to_datetime('2022-12-29 23:30:00')))]# |  ((df['date_time']>pd.to_datetime('2022-06-16 00:00:00')) & (df['date_time']<pd.to_datetime('2022-12-28 23:30:00'))) ]     


    return [df,w_current_level_columns]



@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
