import pandas as pd
from datetime import datetime, timedelta
import pickle
import json
import ast
import numpy as np
from shapely.geometry import Point, LineString
from IRLconstants import DATA_PATH, MODELS_PATH, BERTH_FILE, ANCHOR_FILE, SHIP_FILE, OPERATORS_FILE, VOYAGES_FILE, MIN_BERTH_TIME, MIN_ANCHOR_TIME, HOURS, TIMEFRAMES_PER_DAY, INCOMING_DAYS, EXTRA_FEATURES_NUM

def print_current_time(label):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{label}: {current_time}")

def round_down_to_h_window(dt , hours):
    # Calculate the number of hours to subtract to get to the nearest 3-hour window
    hours_to_subtract = dt.hour % hours
    # Set the minute, second, and microsecond to zero
    rounded_dt = dt.replace(minute=0, second=0, microsecond=0)
    # Subtract the necessary hours to get to the previous 3-hour window
    rounded_dt -= timedelta(hours=hours_to_subtract)
    return rounded_dt

def get_week_of_month(dt):
    first_day_of_month = dt.replace(day=1)
    dom = dt.day
    adjusted_dom = dom + first_day_of_month.weekday()
    return (adjusted_dom - 1) // 7 + 1

# Function to convert string representation of a tuple to an actual tuple of floats
def str_to_float_tuple(coord_str):
    return tuple(map(float, ast.literal_eval(coord_str)))

# Function to calculate distance from a point to a line
def calculate_distances(row, spots_df):
    point = Point(row['LAT'], row['LON'])
    distances = spots_df['line'].apply(lambda line: point.distance(line))
    return distances


def save_prepared_data(data, data_name , terminal, state_based, prefix = 'irl', data_type = 'pkl'):    
    # Convert tuple keys to strings
    if data_type  == 'json':
        data_str = {str(k): v for k, v in data.items()}
        with open(f'{MODELS_PATH}/{prefix}_{data_name}_{terminal}_{state_based}.{data_type}', 'w') as f:
            json.dump(data_str, f)
    else: 
        with open(f'{MODELS_PATH}/{prefix}_{data_name}_{terminal}_{state_based}.{data_type}', 'wb') as f:
            pickle.dump(data, f)

def load_prepared_data(data_name, terminal, state_based, prefix = 'irl', data_type = 'pkl'):    
    # Convert tuple keys to strings
    if data_type  == 'json':        
        with open(f'{MODELS_PATH}/{prefix}_{data_name}_{terminal}_{state_based}.{data_type}', 'r') as f:
            data_str = json.load(f)            
            return {eval(k): v for k, v in data_str.items()}
    else: 
        with open(f'{MODELS_PATH}/{prefix}_{data_name}_{terminal}_{state_based}.{data_type}', 'rb') as f:
            return pickle.load(f)

def extract_time_spent(trajectories, time_spent_lookup, state_based, steps=10):
    """
    Extracts the time spent by each ship in wait zones and berths from the trajectories.
    """
    time_spent = {}
    for step in range(len(trajectories)):
        time = trajectories[step][0]['timeframe_start']
        states = trajectories[step][1]
        for ship, state in states.items():
            loc_local = state
            if state_based:
                loc_local = str(loc_local).rsplit('_', 1)[0]

            if str(loc_local).isdigit() or loc_local == '0' or loc_local == '0.0':  # Filter numerical state_id and wait zone (0)
                time_spent_key = (time, ship, loc_local)
                time_spent_value = time_spent_lookup.get(time_spent_key, 0)
                if ship not in time_spent:
                    time_spent[ship] = {}
                if loc_local not in time_spent[ship]:
                    time_spent[ship][loc_local] = 0
                time_spent[ship][loc_local] += time_spent_value + 1  # Increment time spent by 1 hour
    
    return time_spent

def compare_time_spent(actual_trajectories, predicted_trajectories, time_spent_lookup_actual, time_spent_lookup_predicted, state_based):
    actual_time_spent = extract_time_spent(actual_trajectories, time_spent_lookup_actual, state_based)
    predicted_time_spent = extract_time_spent(predicted_trajectories, time_spent_lookup_predicted, state_based)

    comparison_results = []
    for ship in actual_time_spent:
        actual_states = actual_time_spent[ship]
        predicted_states = predicted_time_spent.get(ship, {})
        
        for state, actual_time in actual_states.items():
            #predicted_time = predicted_states.get(state, 0)
            predicted_time = 0
            if predicted_states:
                any_state = next(iter(predicted_states))
                predicted_time = predicted_states[any_state]
            comparison_results.append({
                'ship': ship,
                'state': state,
                'actual_time_spent': actual_time,
                'predicted_time_spent': predicted_time,
                'match': actual_time == predicted_time
            })
    
    return comparison_results

def calculate_occupation(trajectories):
    """
    Calculates the overall occupation of berths and wait zones at each time step.
    """
    occupation = {'berth': {}, 'wait_zone': {}}
    for step in range(len(trajectories)):
        time = trajectories[step][0]['timeframe_start']
        states = trajectories[step][1]
        
        berth_count = 0
        wait_zone_count = 0
        
        for state in states.values():
            if not str(state).startswith('incom') and str(state).startswith('0'):  # Berth states
                berth_count += 1
            elif str(state).startswith('0'):  # Wait zone state
                wait_zone_count += 1
        
        occupation['berth'][time] = berth_count
        occupation['wait_zone'][time] = wait_zone_count
    
    return occupation

def compare_occupation(actual_trajectories, predicted_trajectories, state_based):
    actual_occupation = calculate_occupation(actual_trajectories)
    predicted_occupation = calculate_occupation(predicted_trajectories)

# Get all time points from both occupations
    all_times = sorted(set(actual_occupation['berth'].keys()).union(predicted_occupation['berth'].keys()))
    
    # Prepare data for DataFrame
    data = []
    for time in all_times:
        actual_berth = actual_occupation['berth'].get(time, 0)
        actual_wait = actual_occupation['wait_zone'].get(time, 0)
        predicted_berth = predicted_occupation['berth'].get(time, 0)
        predicted_wait = predicted_occupation['wait_zone'].get(time, 0)
        
        data.append({
            'time': time,
            'actual_berth_occupation': actual_berth,
            'actual_wait_occupation': actual_wait,
            'predicted_berth_occupation': predicted_berth,
            'predicted_wait_occupation': predicted_wait
        })
    
    # Create DataFrame
    occupation_df = pd.DataFrame(data)
        
    
    return {
        'actual_occupation': actual_occupation,
        'predicted_occupation': predicted_occupation
    }, occupation_df


def calculate_accuracy(trajectories, predictions, state_based):
    total = 0
    cool = 0
    any_berth = 0
    # Extract actions from actuals and predictions
    for step in range(len(trajectories)):
        time = trajectories[step][0]['timeframe_start']
        actions = trajectories[step][2]
        #print(f"in accuracy {actions}")
        predicted_actions = predictions[step][2]
        #print(f"in accuracy predicted {predicted_actions}")
        for ship, action in actions.items():
            print(f"{ship} and {action}")
            if not action.startswith('move_closer_to_'):
                if ship in predicted_actions:
                    if predicted_actions[ship] == action:
                        cool += 1        
                    else:
                        print(f"ship {ship} has predicted action {predicted_actions[ship]} while actual is {action}, total is {total}")
                        if predicted_actions[ship].startswith('stay_at_berth') and action.startswith('stay_at_berth'):
                            any_berth += 1
                        if predicted_actions[ship].startswith('move_to_berth') and action.startswith('move_to_berth'):
                            any_berth += 1

                else:
                    print(f"ship {ship} is not in predicted actions while actual action is {action}, total is {total}")
                total += 1
    return cool/total if total > 0 else 0, (any_berth + cool)/total if total > 0 else 0


def print_traj(trajectories, predictions):
    for step in range(len(trajectories)):
        print(f"step# {step}")
        print(trajectories[step])
        print(f"prediction")
        print(predictions[step])
print('-------------------')


def list_values_for_key_part(dictionary, key_part):
    return [value for key, value in dictionary.items() if key[0] == key_part]
def extract_remaining_parts(dictionary, key_part):
    result = []
    for key in dictionary.keys():
        if key[0] == key_part:
            remaining_parts = key[1:]  # Extract all parts of the key except the first
            result.append(remaining_parts)
    return result

def extract_remaining_parts_2(dictionary, key_part1, key_part2):
    result = []
    for key in dictionary.keys():
        if key[0] == key_part1:
            #remaining_parts = key[1:]  # Extract all parts of the key except the first
            if key[1] == key_part2:
                result.append(key[2])
    return result, [value for key, value in dictionary.items() if key[0] == key_part1 and key[1] == key_part2]


def row_to_dict(row):
    try:
        return {val: col for col, val in row.items() if pd.notna(val)}
    except: 
        print(f"row caused an error {row}")
        raise ValueError('A very specific bad thing happened.')


        
def generate_df_time_window(start_date, end_date, hours):
    # Create a range of datetime values from start_date to end_date with a step of 'hours'
    date_range = pd.date_range(start=start_date, end=end_date, freq=f'{hours}H', closed='left')
    
    # Create a DataFrame from the date range
    df = pd.DataFrame(date_range, columns=['datetime'])
    
    return df


def get_time_features(dt):
    """Returns time-based features for a given datetime."""
    return [
        dt.dayofweek,
        dt.day,
        (dt.day - 1) // 7 + 1,    # Week of the month
        dt.isocalendar()[1]       # ISO week number
    ]

def log_transform(matrix, epsilon=1e-10):
    matrix = np.where(matrix <= 0, epsilon, matrix)  # Replace non-positive values with a small number
    log_transformed_matrix = np.log(matrix)
    return log_transformed_matrix

def stable_softmax(values, threshold = -1000, temperature = 1.0, printVals = False):
    # Ensure values are a numpy array and have the right dtype
    values = np.asarray(values, dtype=np.float64)
    
    # Step 1: Find the maximum value for numerical stability
    max_value = np.max(values)
    
    # Step 2: Shift values by subtracting max_value
    shifted_values = (values - max_value)/temperature
    shifted_values = values/temperature
    # Step 3: Compute the exponentials of the shifted values
    shifted_values = np.clip(shifted_values, threshold, -threshold)  # Avoid extreme values
    #if shifted_values != values/temperature:
    #    print(f"was before clipping {values/temperature}")
    #exp_values = np.exp(shifted_values)
    exp_values = np.clip(np.exp(shifted_values), threshold, None)
    #exp_values = np.where(shifted_values < threshold, 0.0, np.exp(shifted_values)) #np.exp(shifted_values)
    
    # Step 4: Compute the sum of the exponentials
    total = np.sum(exp_values)
    
    # Step 5: Compute the probabilities by normalizing the exponentials
    prob = exp_values / total
    
    prob = np.nan_to_num(prob, nan=0.0)
    if printVals:
        print(f"shifted_values is {shifted_values}")
        print(f"total is {total}")
        print(f"prob is {prob}")
    return prob

def one_hot_to_action(one_hot, int_to_action):
    action_index = np.argmax(one_hot)  # Find the index of the maximum (which is 1)
    return int_to_action[action_index]

def action_to_one_hot(action, action_to_int, num_actions):
    one_hot = np.zeros(num_actions)  # Create a zero vector of length num_actions
    one_hot[action_to_int[action]] = 1  # Set the position corresponding to the action to 1
    return one_hot