#import files; timeframe them 
import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from scipy.special import logsumexp
from datetime import datetime, timedelta
from shapely.geometry import Point, LineString
import time
import hashlib
import sys
from IRLconstants import DATA_PATH, MODELS_PATH, BERTH_FILE, ANCHOR_FILE, SHIP_FILE, OPERATORS_FILE, VOYAGES_FILE, MIN_BERTH_TIME, MIN_ANCHOR_TIME, HOURS, TIMEFRAMES_PER_DAY, INCOMING_DAYS, EXTRA_FEATURES_NUM, TIME_FEATURES, NUM_ITERATIONS, LEARNING_RATE, START_DATE, END_DATE, L1_REG, L2_REG, SOFTMAX_THRESHOLD, TEMPERATURE, STAY_MULTIPLICATOR, NOISE_MEAN, NOISE_SIGMA, USE_LOG, ZERO_TO_NEGATIVE, REMOVE_WRONG_ACTIONS, SCALE_FEATURES
from Utils import print_current_time, round_down_to_h_window, get_week_of_month, str_to_float_tuple, calculate_distances, save_prepared_data, load_prepared_data, extract_time_spent, compare_time_spent, print_traj, calculate_accuracy, compare_occupation, calculate_occupation, extract_remaining_parts, extract_remaining_parts_2, list_values_for_key_part, row_to_dict, generate_df_time_window, get_time_features, stable_softmax, log_transform, one_hot_to_action, action_to_one_hot
from IRL_env_def import define_actions, allowed_actions, action_to_state, define_states, get_next_state
from dateutil.relativedelta import relativedelta
from pandas.errors import SettingWithCopyWarning
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


SPOTS_FILE_SUFFIX = f"SpotsDict_sector.csv"
#f"{DATA_PATH}/MaherSpotsDict_sector.csv"

scaler = StandardScaler()

possible_actions = None
ships_features = None
route_origin_length = None
berth_num = None
wait_num = None
inc_num = None
num_actions = 0
max_list = None
action_to_int = None
# Create number -> action mapping (reverse mapping)
int_to_action = None

#ships to operators assignments (including dates)
def make_ships(p_file_ship, p_file_operators):
    ship = pd.read_csv(p_file_ship)
    ship = ship[['IMO Number', 'size_group','Name', 'Grouped_Operator']]
    ship_ops = pd.read_csv(p_file_operators)
    ship_ops = ship_ops[['IMO','Old_group','New_group','from_date', 'Date Changed']]
    
    ship_ops['Date Changed'] = pd.to_datetime(ship_ops['Date Changed'])

    # Select necessary columns
    filtered_ship_ops = ship_ops[['IMO', 'Old_group', 'from_date', 'Date Changed']]

    # Identify the last 'New_group' for each 'IMO'
    last_new_group = ship_ops.loc[ship_ops.groupby('IMO')['Date Changed'].idxmax()][['IMO', 'New_group', 'Date Changed']]

    # Rename 'New_group' to 'Old_Group' and set 'from_date' to 'Date Changed'
    last_new_group = last_new_group.rename(columns={'New_group': 'Old_group'})
    last_new_group['from_date'] = last_new_group['Date Changed']

    # Concatenate the dataframes
    ship_ops = pd.concat([filtered_ship_ops, last_new_group], ignore_index=True)

    
    merged_df = pd.merge(ship, ship_ops, left_on='IMO Number', right_on='IMO', how='left')
    merged_df = merged_df[['IMO Number', 'size_group','Name','Old_group','from_date', 'Date Changed', 'Grouped_Operator']]
    merged_df['ops'] = merged_df.apply(lambda row: row['Old_group'] if pd.notna(row['Old_group']) else row['Grouped_Operator'], axis=1)
    
    # Fill missing 'start_date' with '01/01/1997'
    merged_df['from_date'] = merged_df['from_date'].fillna('01/01/1997')

    # Convert 'start_date' to datetime
    merged_df['from_date'] = pd.to_datetime(merged_df['from_date'])

    # Convert 'end_date' to datetime and subtract one month
    merged_df['Date Changed'] = pd.to_datetime(merged_df['Date Changed'], errors='coerce')
    merged_df['Date Changed'] = merged_df['Date Changed'].apply(lambda x: x - pd.DateOffset(months=1) if pd.notna(x) else pd.to_datetime('01/01/2100', format='%d/%m/%Y'))
    
    return merged_df.copy()

def transpose_spots(df, max_spot_count):

    # Sort the DataFrame
    df_sorted = df.sort_values(by=['timeframe_start', 'spot_id', 'IMO']).reset_index(drop=True)

    # Initialize previous assignments dictionary
    previous_assignments = {}

    def assign_spots_with_empty_handling(timeframe_df, previous_assignments, max_count_spot_id):
        spot_assignment = {}

        for spot_id, group in timeframe_df.groupby('spot_id'):
            # Get previous assignment for this spot_id
            prev_assignment = previous_assignments.get(spot_id, [None] * max_count_spot_id)
            
            # Track current IMOs
            current_imos = group['IMO'].tolist()
            current_assignment = prev_assignment.copy()

            # Remove IMO from previous assignment if it is still in the current IMOs
            for idx, prev_imo in enumerate(prev_assignment):
                if prev_imo in current_imos:
                    current_assignment[idx] = prev_imo
                    current_imos.remove(prev_imo)
                else:
                    current_assignment[idx] = None  # Mark it as empty if IMO is no longer present
            
            # Fill empty spots with new IMOs
            for idx in range(max_count_spot_id):
                if current_assignment[idx] is None and current_imos:
                    current_assignment[idx] = current_imos.pop(0)

            # Store the assignment
            spot_assignment[spot_id] = current_assignment

        return spot_assignment

    # Iterate over each timeframe and apply the spot assignment
    final_assignments = []

    for timeframe, group in df_sorted.groupby(['timeframe_start']):
        # Assign spots based on previous assignments
        assignment = assign_spots_with_empty_handling(group, previous_assignments, max_spot_count)

        # Convert the assignment dictionary to a DataFrame
        assignment_df = pd.DataFrame({
            'timeframe_start': timeframe,
            'spot_id': list(assignment.keys()),
            **{f'{spot_id}_{i+1}': [spots[i] if i < len(spots) else None for spots in assignment.values()] 
               for i in range(max_spot_count)}  # Create columns for each spot with IMO values
        })

        final_assignments.append(assignment_df)
        previous_assignments = assignment  # Update the previous assignments    # Combine the final assignments into a DataFrame
    final_df = pd.concat(final_assignments).sort_values(by=['timeframe_start', 'spot_id']).reset_index(drop=True)

    return final_df


def transpose_preserve_order(df, max_spot_count):
    # Initialize a dictionary to keep track of IMO positions for each spot_id
    imo_positions = {}
    
    # Initialize a list to store the final transposed data
    final_transposed_data = []
    
    # Process each timeframe_start
    for timeframe in df['timeframe_start'].unique():
        # Filter data for the current timeframe_start
        current_data = df[df['timeframe_start'] == timeframe]
        
        # Initialize a dictionary to store the transposed row
        transposed_row = {'timeframe_start': timeframe}
        
        # Process each spot_id
        for spot in current_data['spot_id'].unique():
            # Get the IMOs for the current spot_id and timeframe
            imos = current_data[current_data['spot_id'] == spot]['IMO'].tolist()
            
            # Reset positions for the current spot_id and timeframe
            last_position = 0
            
            # Update the IMO positions
            for i in range(len(imos)):
                column_name = f'{spot}_{last_position + 1}'
                transposed_row[column_name] = imos[i]
                last_position += 1
            
            # Fill in empty spots if needed
            for j in range(len(imos), max_spot_count):
                column_name = f'{spot}_{last_position + 1}'
                transposed_row[column_name] = None
                last_position += 1
        
        # Append the transposed row to the list
        final_transposed_data.append(transposed_row)
    
    # Convert the list of dictionaries to a DataFrame
    transposed_df = pd.DataFrame(final_transposed_data)
    
    return transposed_df

def extract_trajectories_state_based(berth, anchor, incoming, hours, terminal, state_based):
    # Convert spot_id to int in berth
    # now state is formed by number of berth (max spot_id), max at waiting_zone and max at incoming_n (eventually, all of them will be there) * all incoming states
    global max_wait
    global wait_num
    global inc_num
    berth_copy = berth.copy()
    berth_copy['spot_id'] = pd.to_numeric(berth_copy['spot_id'], errors='coerce')  # Convert to numeric, NaN if not possible
    berth_copy = berth_copy.dropna(subset=['spot_id'])  # Drop rows where 'spot_id' is NaN
    max_berth = berth_copy['spot_id'].astype(int).max()

    wait_counts = anchor.groupby('timeframe_start')['IMO'].count()
    print(f"wait_counts is {wait_counts}")
    max_wait_count = wait_counts.max()
    save_prepared_data(max_wait_count, 'max_wait', terminal, state_based) 
    max_wait = max_wait_count
    wait_num = max_wait_count
    print(f"max_wait_count is {max_wait_count}")
    inc_counts = incoming.groupby(['timeframe_start', 'spot_id'])['IMO'].count()
    save_prepared_data(inc_counts, 'inc_num', terminal, state_based) 
    inc_num = inc_counts.max()
    print(f"wait_counts is {inc_counts}")
    max_inc_count = inc_counts.max()
    total_states = max_berth + wait_counts + inc_counts * INCOMING_DAYS * TIMEFRAMES_PER_DAY
    print(f"max_inc_count is {max_inc_count}")

    # Initialize the list to store trajectories
    trajectories = []
    
    #berth_transposed = transpose_spots(berth.copy(), 1)
    #anchor_transposed = transpose_spots(anchor.copy(), max_wait_count)
    #incoming_transposed = transpose_spots(incoming.copy(), max_inc_count)
    berth_transposed = transpose_preserve_order(berth.copy(), 1)
    anchor_transposed = transpose_preserve_order(anchor.copy(), max_wait_count)
    incoming_transposed = transpose_preserve_order(incoming.copy(), max_inc_count)

    merged_df = pd.merge(berth_transposed, anchor_transposed, on='timeframe_start', how='outer')
    merged_df = pd.merge(merged_df, incoming_transposed, on='timeframe_start', how='outer')	
    
    # Combine dataframes to get all unique timeframes
    all_unique_timeframes = sorted(berth['timeframe_start'].unique())
    
    # Iterate over each unique timeframe
    for start_time in all_unique_timeframes[:-1]:
        #current_time = start_time
        current_time = pd.to_datetime(start_time)
        next_time = current_time + pd.Timedelta(hours=hours)  # Using the passed hours parameter
        
        # State dictionaries for current and next timeframe
        current_state = {}
        next_state = {}
        action_state = {}
        warnings.filterwarnings('ignore', category=SettingWithCopyWarning)
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        # Populate current_state from berth, anchor, and incoming dataframes
        berth_current = berth[(berth['timeframe_start'] == current_time) & (berth['IMO'].notna())]
        berth_current.loc[:, 'spot_id'] = berth_current['spot_id'].astype(int)
        #berth_current['spot_id'] = berth_current['spot_id'].astype(int)
        anchor_current = anchor[(anchor['timeframe_start'] == current_time) & (anchor['IMO'].notna())]
        incoming_current = incoming[incoming['timeframe_start'] == current_time]

        berth_next = berth[(berth['timeframe_start'] == next_time) & (berth['IMO'].notna())]
        berth_next['spot_id'] = berth_next['spot_id'].astype(int)
        anchor_next = anchor[(anchor['timeframe_start'] == next_time) & (anchor['IMO'].notna())]
        incoming_next = incoming[incoming['timeframe_start'] == next_time]

        # Combine current state data
        for df in [berth_current, anchor_current, incoming_current]:
            for _, row in df.iterrows():
                current_state[row['IMO']] = row['spot_id']
        # Determine actions based on current state and next state
        for imo in current_state:
            action = 'leave_system'  # Default action
            if imo in berth_current['IMO'].values:
                if imo in berth_next['IMO'].values:
                    action = f"stay_at_berth_{current_state[imo]}"
                else:
                    action = 'leave_system'
            elif imo in anchor_current['IMO'].values:
                if imo in berth_next['IMO'].values:
                    action = f"move_to_berth_{berth_next[berth_next['IMO'] == imo]['spot_id'].values[0]}"
                elif imo in anchor_next['IMO'].values:
                    action = 'stay_at_wait_zone'
                else:
                    action = 'leave_system'
            elif imo in incoming_current['IMO'].values:
                if imo in berth_next['IMO'].values:
                    action = f"move_to_berth_{berth_next[berth_next['IMO'] == imo]['spot_id'].values[0]}"
                elif imo in anchor_next['IMO'].values:
                    action = 'move_to_wait_zone'
                elif imo in incoming_next['IMO'].values:
                    action = f"move_closer_to_{incoming_next[incoming_next['IMO'] == imo]['spot_id'].values[0]}"
                else:
                    action = 'leave_system'

            # Append the action to the action_state dictionary
            action_state[imo] = action
        current_state_dict = row_to_dict(merged_df[merged_df['timeframe_start'] == current_time].drop('timeframe_start', axis = 1).iloc[0])
        next_state_dict = row_to_dict(merged_df[merged_df['timeframe_start'] == next_time].drop('timeframe_start', axis = 1).iloc[0])
        temporal_info = {
            'timeframe_start': current_time,
            'day_of_week': current_time.dayofweek,
            'day_of_month': current_time.day,
            'week_of_month': (current_time.day - 1) // 7 + 1,
            'week_of_year': current_time.isocalendar()[1]
        }                
        # Append the (state, action, next_state) tuple to trajectories
        trajectories.append((temporal_info, current_state_dict, action_state, next_state_dict))
    return trajectories        



def check_criteria(row, df1):
    imo = row['IMO']
    end_date_plus_one = row['end_time'] + pd.Timedelta(days=1)
    return df1[(df1['IMO'] == imo) & (df1['start_time'] > row['end_time']) & (df1['start_time'] <= end_date_plus_one)].shape[0] > 0


def get_timeframed(df, is_spot = True, hours = 3):
    
    def generate_timeframes(row):
        timeframes = []
        current_time = row['start_time']
        while current_time < row['end_time']:
            next_time = min(current_time + pd.Timedelta(hours=hours), row['end_time'])
            spot_id = 0
            if is_spot:
                spot_id = row['spot_id']
            timeframe = {
                'IMO': row['IMO'],
                'timeframe_start': current_time,
                'timeframe_end': next_time,
                'hours_spent': (next_time - current_time).total_seconds() / 3600,
                'spot_id': spot_id
            }
            timeframes.append(timeframe)
            current_time = next_time
        return timeframes

    # Function to generate fixed 6-hour timeframes for a given day
    def generate_fixed_timeframes(start_date, end_date):
        fixed_timeframes = []
        current_time = start_date
        while current_time <= end_date:
            fixed_timeframes.append(current_time)
            current_time += pd.Timedelta(hours=hours)
        return fixed_timeframes

    # Function to assign time intervals to fixed timeframes
    def assign_to_timeframes(row):
        intervals = []
        for i in range(len(fixed_timeframes) - 1):
            start_frame = fixed_timeframes[i]
            end_frame = fixed_timeframes[i + 1]
            if row['start_time'] < end_frame and row['end_time'] > start_frame:
                start_interval = max(row['start_time'], start_frame)
                end_interval = min(row['end_time'], end_frame)
                hours_spent = (end_interval - start_interval).total_seconds() / 3600
                spot_id = 0 
                if is_spot:
                    spot_id = row['spot_id'] 
                intervals.append({
                    'IMO': row['IMO'],
                    'timeframe_start': start_frame,
                    'hours_spent': hours_spent,
                    'start_time': row['start_time'],
                    'end_time': row['end_time'],
                    'spot_id': spot_id
                })
        return intervals
    df_sub = df.copy()
    # Generate fixed 6-hour timeframes for the given date range
    df_sub['start_time'] = pd.to_datetime(df_sub['start_time'])
    df_sub['end_time'] = pd.to_datetime(df_sub['end_time'])
    start_date = df_sub['start_time'].min().normalize()
    end_date = df_sub['end_time'].max().normalize() + pd.Timedelta(days=1)
    fixed_timeframes = generate_fixed_timeframes(start_date, end_date)


    # Apply the function and explode the list of dictionaries into a DataFrame
    timeframes = df_sub.apply(assign_to_timeframes, axis=1)
    #timeframes = df_sub.apply(generate_timeframes, axis=1)
    timeframes = [item for sublist in timeframes for item in sublist]
    df_timeframes = pd.DataFrame(timeframes)

    # Apply the function and explode the list of dictionaries into a DataFrame
    # Group by timeframe_start and sum the hours_spent for all vehicles
    grouped_df = df_timeframes.groupby('timeframe_start').agg({'hours_spent': 'sum'}).reset_index()

    # Find the maximum sum of hours spent per timeframe
    max_hours_spent = grouped_df['hours_spent'].max()

    # Get the timeframes with the maximum hours spent
    max_hours_df = grouped_df[grouped_df['hours_spent'] == max_hours_spent]
    return df_timeframes

def fill_tf_gaps(df, hours = 3, remove_less_half = True):
    # Remove records with duration less than half of the timeframe
    if remove_less_half:
        #df = df[df['hours_spent'] >= (hours / 2)]
        df = df[~((df['hours_spent'] < hours) & (df['start_time'] == df['timeframe_start']))]

    # Generate all time windows from 2 Jan 2015 till 30 Sep 2023
    start_date = pd.Timestamp('2015-01-02 00:00')
    end_date = pd.Timestamp('2023-09-30 23:59')
    all_timeframes = pd.date_range(start=start_date, end=end_date, freq=f'{hours}H')

    # Create a complete dataframe with all time windows
    complete_df = pd.DataFrame(all_timeframes, columns=['timeframe_start'])

    # Merge original dataframe with the complete time windows dataframe on 'frame_start_time'
    merged_df = pd.merge(complete_df, df, left_on='timeframe_start', right_on='timeframe_start', how='left')
    return merged_df.copy()

# Iterate through the merged data and apply conditions
def clear_or_delete(row, anchor_tf_en):
    imo = row['IMO']
    frame_start_time = row['timeframe_start']
    if not anchor_tf_en[(anchor_tf_en['timeframe_start'] == frame_start_time) & (anchor_tf_en['IMO'] == imo)].empty:
        return True #False
    #elif anchor_tf_en[(anchor_tf_en['timeframe_start'] == frame_start_time)].shape[0] > 1:
    #    return False
    else:
        return False

# Function to find the earliest position of each segment of consecutive timeframes
def find_earliest_positions(df, hours=3):
    result = []
    for imo, group in df.groupby('IMO'):
        group = group.sort_values('timeframe_start').reset_index(drop=True)
        segment_start = group.loc[0, 'timeframe_start']
        
        for i in range(1, len(group)):
            current_time = group.loc[i, 'timeframe_start']
            previous_time = group.loc[i - 1, 'timeframe_start']
            
            if current_time != previous_time + pd.Timedelta(hours=hours):
                result.append((imo, segment_start))
                segment_start = current_time
        
        result.append((imo, segment_start))  # Ensure the last segment is added
    return result


# Function to generate incoming dataframe
def generate_incoming_dataframe(result_df, hours, n, k):
    incoming_data = []
    
    for _, row in result_df.iterrows():
        imo = row['IMO']
        start_time = row['timeframe_start']
        
        for i in range(n * k):
            timeframe_i = start_time - pd.Timedelta(hours=hours * (i + 1))
            incoming_data.append({
                'timeframe_start': timeframe_i,
                'IMO': imo,
                f'spot_id': f'incoming_{i+1}'
            })
    
    incoming_df = pd.DataFrame(incoming_data)
    return incoming_df

def extract_trajectories(berth, anchor, incoming, hours):
    # Convert spot_id to int in berth
    

    # Initialize the list to store trajectories
    trajectories = []
    
    # Combine dataframes to get all unique timeframes
    all_unique_timeframes = sorted(berth['timeframe_start'].unique())
    
    # Iterate over each unique timeframe
    for start_time in all_unique_timeframes:
        #current_time = start_time
        current_time = pd.to_datetime(start_time)
        next_time = current_time + pd.Timedelta(hours=hours)  # Using the passed hours parameter
        
        # State dictionaries for current and next timeframe
        current_state = {}
        next_state = {}
        action_state = {}

        # Populate current_state from berth, anchor, and incoming dataframes
        berth_current = berth[(berth['timeframe_start'] == current_time) & (berth['IMO'].notna())]
        berth_current['spot_id'] = berth_current['spot_id'].astype(int)
        anchor_current = anchor[(anchor['timeframe_start'] == current_time) & (anchor['IMO'].notna())]
        incoming_current = incoming[incoming['timeframe_start'] == current_time]

        berth_next = berth[(berth['timeframe_start'] == next_time) & (berth['IMO'].notna())]
        berth_next['spot_id'] = berth_next['spot_id'].astype(int)
        anchor_next = anchor[(anchor['timeframe_start'] == next_time) & (anchor['IMO'].notna())]
        incoming_next = incoming[incoming['timeframe_start'] == next_time]

        # Combine current state data
        for df in [berth_current, anchor_current, incoming_current]:
            for _, row in df.iterrows():
                current_state[row['IMO']] = row['spot_id']

        # Determine actions based on current state and next state
        for imo in current_state:
            action = 'leave_system'  # Default action
            if imo in berth_current['IMO'].values:
                if imo in berth_next['IMO'].values:
                    action = f"stay_at_berth_{current_state[imo]}"
                    if str(current_state[imo]).startswith("incoming"):
                        print(f"for IMO {imo} we passed both berth_current and berth_next, but it should be {berth[(berth['timeframe_start'] == current_time) & (berth['IMO'].notna())]}")
                else:
                    action = 'leave_system'
            elif imo in anchor_current['IMO'].values:
                if imo in berth_next['IMO'].values:
                    action = f"move_to_berth_{berth_next[berth_next['IMO'] == imo]['spot_id'].values[0]}"
                elif imo in anchor_next['IMO'].values:
                    action = 'stay_at_wait_zone'
                else:
                    action = 'leave_system'
            elif imo in incoming_current['IMO'].values:
                if imo in berth_next['IMO'].values:
                    action = f"move_to_berth_{berth_next[berth_next['IMO'] == imo]['spot_id'].values[0]}"
                elif imo in anchor_next['IMO'].values:
                    action = 'move_to_wait_zone'
                elif imo in incoming_next['IMO'].values:
                    action = f"move_closer_to_{incoming_next[incoming_next['IMO'] == imo]['spot_id'].values[0]}"
                else:
                    action = 'leave_system'

            # Append the action to the action_state dictionary
            action_state[imo] = action

        # Populate next_state keeping the order of current_state
        for imo in current_state:
            if imo in berth_next['IMO'].values:
                next_state[imo] = berth_next[berth_next['IMO'] == imo]['spot_id'].values[0]
            elif imo in anchor_next['IMO'].values:
                next_state[imo] = anchor_next[anchor_next['IMO'] == imo]['spot_id'].values[0]
            elif imo in incoming_next['IMO'].values:
                next_state[imo] = incoming_next[incoming_next['IMO'] == imo]['spot_id'].values[0]
            else:
                if imo in next_state:
                #next_state[imo] = 'leave_system'
                    del next_state[imo]
        temporal_info = {
            'timeframe_start': current_time,
            'day_of_week': current_time.dayofweek,
            'day_of_month': current_time.day,
            'week_of_month': (current_time.day - 1) // 7 + 1,
            'week_of_year': current_time.isocalendar()[1]
        }                

        # Append the (state, action, next_state) tuple to trajectories
        trajectories.append((temporal_info, current_state, action_state, next_state))
    return trajectories        

def process_ships_dataframe(ships):
    # Ensure 'IMO' values are strings and prepend 'IMO'
    ships['IMO'] = 'IMO' + ships['IMO Number'].astype(str)

    # Select required columns
    ships = ships[['IMO', 'size_group', 'from_date', 'Date Changed', 'ops']]

    # Create dummy variables for 'ops' and 'size_group'
    ops_dummies = pd.get_dummies(ships['ops'], prefix='ops')
    size_group_dummies = pd.get_dummies(ships['size_group'], prefix='size_group')

    # Drop the original 'ops' and 'size_group' columns
    ships = ships.drop(['ops', 'size_group'], axis=1)

    # Concatenate the original dataframe with the new dummy columns
    ships = pd.concat([ships, ops_dummies, size_group_dummies], axis=1)

    return ships

def make_berth(terminal, spots_df):
    berth = pd.read_csv(BERTH_FILE)
    berth = berth[berth['nearestPort'] == terminal]
    berth = berth[berth['TimeSpent'] > MIN_BERTH_TIME]
    berth = berth[['IMO', 'start_time', 'end_time', 'LAT', 'LON']]
    berth['start_time'] = pd.to_datetime(berth['start_time'])
    berth['end_time'] = pd.to_datetime(berth['end_time'] )
    # Calculate distances and find the minimum for each berth
    berth = berth.join(berth.apply(calculate_distances, axis=1, spots_df=spots_df).rename(columns=lambda x: f'distance_to_spot_{x+1}'))
    distance_columns = [col for col in berth.columns if col.startswith('distance_to_spot_')]
    berth['spot_id'] = berth[distance_columns].idxmin(axis=1).apply(lambda x: str(int(x.split('_')[-1])))

    # Drop the distance columns to clean up the dataframe
    berth = berth.drop(columns=distance_columns)
    
    return berth

def make_spots(terminal):
    terminal_cut = f"{terminal}_"
    if terminal == 'NY_Maher':
        terminal_cut = 'Maher'
    SPOTS_FILE = f"{DATA_PATH}/{terminal_cut}SpotsDict_sector.csv"
    spots_df = pd.read_csv(SPOTS_FILE) # have to rewrite to terminal - dependent spot
    # Convert start_point and end_point columns to tuples of floats
    spots_df['start_point'] = spots_df['start_point'].apply(str_to_float_tuple)
    spots_df['end_point'] = spots_df['end_point'].apply(str_to_float_tuple)
    spots_df['line'] = spots_df.apply(lambda row: LineString([row['start_point'], row['end_point']]), axis=1)
    return spots_df

def make_voyages():
    voyages = pd.read_csv(VOYAGES_FILE)
    voyages['IMO'] = 'IMO' + voyages['IMO'].astype(int).astype(str)

    voyages_df = voyages[['IMO','Route_Start', 'Route_End', 'Route_Origin']]
    route_origin_dummies = pd.get_dummies(voyages_df['Route_Origin'], prefix='Route_Origin')

    # Drop the original 'Route_Origin' column
    voyages_df = voyages_df.drop(['Route_Origin'], axis=1)

        # Concatenate the original dataframe with the new dummy columns
    voyages_df = pd.concat([voyages_df, route_origin_dummies], axis=1)
    voyages_df['Route_Start'] = pd.to_datetime(voyages_df['Route_Start'])
    voyages_df['Route_End'] = pd.to_datetime(voyages_df['Route_End'])

    return voyages_df

def make_anchor(port, berth):
    anchor = pd.read_csv(ANCHOR_FILE)
    anchor = anchor[anchor['TimeSpent'] > MIN_ANCHOR_TIME]
    anchor = anchor[anchor['nearestPort'] == port][['IMO','start_time','end_time']]

    anchor['start_time'] = pd.to_datetime(anchor['start_time'])
    anchor['end_time'] = pd.to_datetime(anchor['end_time'])
    # Apply the function to filter df2
    anchor = anchor[anchor.apply(check_criteria, axis=1, args=(berth,))]
    return anchor

global loc_list
def get_loc_list(positions):
    global wait_num
    global inc_num    
    loc_list = []
    for x in positions:
        if str(x).isdigit() and x != '0.0' and x != '0':
            loc_list.append(f"{x}_{1}")
        elif str(x) == '0.0' or x=='0':
            for i in range(wait_num):
                loc_list.append(f"{x}_{i+1}")
        else:
            for i in range(inc_num):
                loc_list.append(f"{x}_{i+1}")
    return loc_list
nc = 1
def extract_features(state, feature_num, time_ef, local_lookup, positions, features, state_based, lookup_debug = False, debug = False):
    global possible_actions
    global ships_features
    global route_origin_length
    global loc_list
    global nc
    #inc_num = 50
    local_features = []
    timeframe_start = time_ef['timeframe_start']
    if state_based:        
        kkk = 0
        loc_list_len = len(loc_list)        
        for x in loc_list:
            #pos = x.rsplit("_", 1)[0]
            pos = x
            ship = None
            for sh, loc in state.items():
                if pos == loc:
                    ship = sh
                    break
            #for pos in positions:
            #    features.append(1 if location == pos else 0)
            if ship != None:
                #for act in possible_actions:
                #    features.append(1 if act == action.get(ship) else 0)
                        # Ship features
                
                ship_key = (timeframe_start, ship)
                ship_features = features.loc[ship_key]
                #features.get(ship_key, [0] * ships_features)
                local_features.extend(ship_features)
                
                # Voyage features9
                #loc_local = pos #str(location)
                # Get time_spent from the lookup dictionary
                #if state_based:
                loc_local = str(pos).rsplit('_', 1)[0]
                time_spent_key = (ship, loc_local, timeframe_start)
                time_spent_value = local_lookup.get(time_spent_key, 0)
                if lookup_debug and str(loc_local).isdigit() and False:
                    print(f"for ship {ship} time spent value is {time_spent_value}, location is {location}, time spent key is {time_spent_key}")
                    #print(extract_remaining_parts_2(local_lookup, ship, location))
                    #print(extract_remaining_parts(local_lookup, ship))
                for k in range(EXTRA_FEATURES_NUM - 1):
                    if k < time_spent_value :
                        local_features.append(k * 10)
                    #elif k > time_spent_value - 1:
                    #    local_features.append(1*STAY_MULTIPLICATOR)
                    else:
                        local_features.append(0)
                #local_features.append(time_spent_value)
                #is_new ship 
                if time_spent_value == 1:
                    local_features.append(1)
                elif time_spent_value == 0: 
                    local_features.append(0)
                else:
                    local_features.append(-1)

            kkk += 1
            if len(local_features) < (feature_num // loc_list_len) * kkk:
                local_features.extend([0] * ((feature_num // loc_list_len) * kkk - len(local_features)))

            #local_lookup = time_spent_lookup
    else:
        for ship, location in state.items():
            for pos in positions:
                local_features.append(1 if location == pos else 0)

            ship_key = (timeframe_start, ship)
            ship_features = features.loc[ship_key] #features.get(ship_key, [0] * ships_features)
            #ero_features = [0] * len(ship_features) 
            local_features.extend(ship_features)
            #ocal_features.extend(zero_features)

            loc_local = str(location)
            # Get time_spent from the lookup dictionary
            time_spent_key = (ship, loc_local, timeframe_start)
            time_spent_value = local_lookup.get(time_spent_key, 0)
            if (lookup_debug) and str(loc_local).isdigit():
                print(f"for ship {ship} time spent value is {time_spent_value}, location is {location}, time spent key is {time_spent_key}")
                #print(extract_remaining_parts_2(local_lookup, ship, location))
                #print(extract_remaining_parts(local_lookup, ship))

            for k in range(EXTRA_FEATURES_NUM - 1):
                if k < time_spent_value :
                    local_features.append(k * 10)
                #elif k > time_spent_value - 1:
                #    local_features.append(1*STAY_MULTIPLICATOR)
                else:
                    local_features.append(0)
            #local_features.append(time_spent_value)
            #is_new ship 
            if time_spent_value == 1:
                local_features.append(1)
            elif time_spent_value == 0: 
                local_features.append(0)
            else:
                local_features.append(-1)

        if len(local_features) < feature_num:
            local_features.extend([0] * (feature_num - len(local_features)))
        #if nc < 30:
        #    np.set_printoptions(threshold=np.inf)
        #    print(f"------------local_features---for ship {ship}--------")
        #    print(np.array(local_features))
        #    np.set_printoptions(threshold=1000)
        #    nc += 1
    return np.array(local_features)

def precompute_features(trajectories, feature_num, time_spent_lookup, positions, features, state_based):
    precomputed_features = {}
    print(f"we have  {len(trajectories)} trajectories")
    k = 0
    debug = False
    for trajectory in trajectories:
        time, state, action, next_state = trajectory
        state_key = time['timeframe_start']
         # Modify next_state in trajectory to remove entries with 'leave_system'
        updated_next_state = {ship: pos for ship, pos in next_state.items() if pos != 'leave_system'}
    
        # Create a new tuple with the updated next_state
        new_trajectory = (time, state, action, updated_next_state)
    
        # Replace the old trajectory in the list with the new one
        trajectories[trajectories.index(trajectory)] = new_trajectory
    
        if state_key not in precomputed_features:
            precomputed_features[state_key] = {}
        
        # Store features for each ship and action separately
        for ship, ship_action in action.items():
            action_key = ship
            if action_key not in precomputed_features[state_key]:
                local_features = extract_features(state, feature_num, {'timeframe_start': time['timeframe_start']}, time_spent_lookup, positions, features, state_based, False, debug)
                precomputed_features[state_key][action_key] = local_features
        k += 1 
        if k % 1000 == 0:
            print_current_time(f"{k} trajectories passed")
    print_current_time(f"trajectories passed returning them")
    return precomputed_features, trajectories

#compute_expert_feature_expectations(trajectories, feature_num, time_spent_lookup, positions, features, state_based)

'''
def compute_expert_feature_expectations(trajectories, feature_size, time_spent_lookup, positions, features, state_based):
    feature_expectations = np.zeros(feature_size)
    for trajectory in trajectories:
        time, state, action, _ = trajectory
        local_features = extract_features(state, feature_size, time, time_spent_lookup, positions, features, state_based)
        #print(f"features{len(local_features)} and {len(feature_expectations)}")
        feature_expectations += local_features
    return feature_expectations / len(trajectories)
'''
def compute_expert_feature_expectations(trajectories, feature_size, time_spent_lookup, positions, features, state_based):
    global num_actions
    global action_to_int
    global int_to_action 
    
    # Initialize matrix to store feature expectations for each action
    feature_expectations = np.zeros((num_actions, feature_size))
    
    for trajectory in trajectories:
        time, state, actions, _ = trajectory  # `actions` is now a list of actions
        local_features = extract_features(state, feature_size, time, time_spent_lookup, positions, features, state_based)
        # Loop over each action in the action list        
        for action in actions:
            action_number = action_to_int.get(action)
            # Accumulate the features for the specific action
            feature_expectations[action_number] += local_features
    
    # Average feature expectations over all trajectories
    return feature_expectations / len(trajectories)

def data_preparation(terminal, port, spots_df, state_based = False, calc_features = False):
    ships = make_ships(SHIP_FILE,OPERATORS_FILE)    
    berth = make_berth(terminal, spots_df)
    anchor = make_anchor(port, berth)
    anchor.to_csv(f"{MODELS_PATH}/anchor.csv")

    # Find the spot_id with the minimum distance for each berth
    berth_tf = get_timeframed(berth, True, HOURS)
    anchor_tf = get_timeframed(anchor, False, HOURS)
    
    berth_tf_en = fill_tf_gaps(berth_tf, HOURS,True)
    anchor_tf_en = fill_tf_gaps(anchor_tf, HOURS,False)
    anchor_tf_en.to_csv(f"{MODELS_PATH}/anchor_tf_en.csv")
    
    # Apply the function to filter the ancor_tf_en dataframe
    anchor_tf_en['to_delete'] = anchor_tf_en.apply(lambda row: clear_or_delete(row, berth_tf_en), axis=1)

    # Clear IMO and duration where necessary
    anchor_tf_en.loc[anchor_tf_en['to_delete'], ['IMO', 'hours_spent']] = [np.nan, np.nan]

    # Remove the 'to_delete' column
    anchor_tf_en.drop(columns=['to_delete'], inplace=True)

    anchor_tf_en.to_csv(f'{MODELS_PATH}/after_clear_And_Delete.csv')########
    #filling gaps (if exit from anchor but didn't arrive to berth)
    a3 = pd.merge(anchor_tf_en, anchor_tf_en, 
                  left_on=['IMO', anchor_tf_en['timeframe_start'] + pd.to_timedelta(HOURS, unit='h')],
                  right_on=['IMO', 'timeframe_start'],
                  suffixes=('_a1', '_a2'),
                  how='left')
    # Select rows from a1 where there are no corresponding records in a2
    a3 = a3[a3['timeframe_start_a2'].isnull()]

    # Step 2: Join with berth_df_en to find final result
    final_result = pd.merge(a3, berth_tf_en,
                        left_on=['IMO', a3['timeframe_start_a1'] + pd.to_timedelta(HOURS, unit='h')],
                        right_on=['IMO', 'timeframe_start'],
                        suffixes=('_a3', '_b1'),
                        how='left')

    # Select rows from a3 where there are no corresponding records in b1
    final_result = final_result[final_result['timeframe_start_b1'].isnull()]
    a3 = a3[a3['timeframe_start_a2'].isnull()]

    # Step 2: Join with berth_tf_en to find final result
    final_result = pd.merge(a3, berth_tf_en,
                            left_on=['IMO', a3['timeframe_start_a1'] + pd.to_timedelta(HOURS, unit='h')],
                            right_on=['IMO', 'timeframe_start'],
                            suffixes=('_a3', '_b1'),
                            how='left')
    final_result = final_result[final_result['timeframe_start_b1'].isnull()]

    # Step 3: Remove rows from anchor_tf_en or leave one empty record if no other IMO
    # Remove these rows from anchor_tf_en
    rows_to_remove = final_result[['IMO', 'timeframe_start_a1']]
    rows_to_remove.columns = ['IMO', 'timeframe_start']

    anchor_tf_en = anchor_tf_en.merge(rows_to_remove, on=['IMO', 'timeframe_start'], how='left', indicator=True)
    anchor_tf_en = anchor_tf_en[anchor_tf_en['_merge'] == 'left_only'].drop(columns=['_merge'])

    # Step 4: Ensure at least one empty record for timeframes with no other IMOs
    missing_timeframes = final_result[['timeframe_start_a1']].drop_duplicates()
    missing_timeframes.columns = ['timeframe_start']
    missing_timeframes = missing_timeframes[~missing_timeframes['timeframe_start'].isin(anchor_tf_en['timeframe_start'])]

    # Create empty records for missing timeframes
    empty_records = pd.DataFrame({
        'IMO': [None] * len(missing_timeframes),
        'timeframe_start': missing_timeframes['timeframe_start'],
        'hours_spent': [None] * len(missing_timeframes)
    })
    anchor_tf_en = pd.concat([anchor_tf_en, empty_records], ignore_index=True)

    # Combine both DataFrames
    combined_df = pd.concat([anchor_tf_en.assign(dataset='anchor'), berth_tf_en.assign(dataset='berth')])
    combined_df = combined_df.sort_values(by=['IMO', 'timeframe_start']).reset_index(drop=True)
    # Find earliest positions for each segment
    earliest_positions = find_earliest_positions(combined_df, HOURS)

    # Create a DataFrame from the result
    result_df = pd.DataFrame(earliest_positions, columns=['IMO', 'timeframe_start'])

    # Generate incoming dataframe
    incoming_df = generate_incoming_dataframe(result_df, HOURS, TIMEFRAMES_PER_DAY, INCOMING_DAYS)
    
    ships_df = process_ships_dataframe(ships)
    ships_df['from_date'] = pd.to_datetime(ships_df['from_date'])
    ships_df['Date Changed'] = pd.to_datetime(ships_df['Date Changed'])    

    
    voyages_df = make_voyages()
    
    berth_tf_en_copy = berth_tf_en.copy()
    berth_tf_en_copy['spot_id'] = pd.to_numeric(berth_tf_en_copy['spot_id'], errors='coerce').astype('Int64')  # Use 'Int64' to handle NaN

    combined_df = pd.concat([berth_tf_en_copy[['timeframe_start', 'IMO', 'spot_id']],
                             anchor_tf_en[['timeframe_start', 'IMO', 'spot_id']],
                             incoming_df[['timeframe_start', 'IMO', 'spot_id']]])

    # Sort the DataFrame by IMO, spot_id, and timeframe_start
    combined_df_tf = combined_df.sort_values(by=['IMO', 'spot_id', 'timeframe_start']).reset_index(drop=True)

    # Shift the DataFrame to compare current row with previous row
    combined_df_tf['prev_timeframe_start'] = combined_df_tf.groupby(['IMO', 'spot_id'])['timeframe_start'].shift(1)

    # Calculate the difference in days between current and previous row
    combined_df_tf['time_diff'] = (combined_df_tf['timeframe_start'] - combined_df_tf['prev_timeframe_start']).dt.total_seconds() / 3600

    # Identify the start of a new sequence (where difference is more than expected or it's the first row)
    combined_df_tf['new_sequence'] = (combined_df_tf['time_diff'] > HOURS) | (combined_df_tf['time_diff'].isna())

    # Calculate cumulative sum of new_sequence to create a group id for each sequence
    combined_df_tf['sequence_id'] = combined_df_tf.groupby(['IMO', 'spot_id'])['new_sequence'].cumsum()

    # Calculate the time_spent within each sequence
    combined_df_tf['time_spent'] = combined_df_tf.groupby(['IMO', 'spot_id', 'sequence_id']).cumcount() + 1
    combined_df_tf.to_csv(f'{MODELS_PATH}/combined_tf_new.csv')
    grouped_df = combined_df_tf.groupby(['IMO', 'spot_id', 'timeframe_start'])['time_spent'].max().reset_index()
    grouped_df['spot_id'] = grouped_df['spot_id'].astype(str)
# Now convert this grouped DataFrame to a dictionary
    time_spent_lookup = grouped_df.set_index(['IMO', 'spot_id', 'timeframe_start'])['time_spent'].to_dict()
    # Drop unnecessary columns
    
    precomputed_ship_features = {}
    for ship in ships_df['IMO'].unique():
        ship_records = ships_df[ships_df['IMO'] == ship]
        for _, record in ship_records.iterrows():
            for month in range(record['from_date'].month, record['Date Changed'].month + 1):
                key = (ship, month, record['from_date'].year, record['Date Changed'].year)
                precomputed_ship_features[key] = [record[col] for col in sorted(ships_df.columns) if col.startswith('ops') or col.startswith('size_group')]
    #first_voyage_features = next(iter(precomputed_voyage_features.values()))
    route_origin_columns = [col for col in voyages_df.columns if col.startswith('Route_Origin')]
    # Precompute voyage features
    precomputed_voyage_features = {}
    for ship in voyages_df['IMO'].unique():
        voyage_records = voyages_df[voyages_df['IMO'] == ship]
        for _, record in voyage_records.iterrows():
            for month in range(record['Route_Start'].month, record['Route_End'].month + 1):
                key = (ship, month, record['Route_Start'].year, record['Route_End'].year)
                precomputed_voyage_features[key] = [record.get(col, 0) for col in route_origin_columns]    
    trajectories = ()
    if state_based:
        trajectories = extract_trajectories_state_based(berth_tf_en, anchor_tf_en, incoming_df, HOURS, terminal, state_based)
    else:
        trajectories = extract_trajectories(berth_tf_en, anchor_tf_en, incoming_df, HOURS)
    print(f"traj is {trajectories[300]}")
    return trajectories, combined_df_tf, time_spent_lookup, precomputed_ship_features, precomputed_voyage_features

def feature_lookup_preparation(terminal, port, spots_df, state_based = False, calc_features = False):
    ships = make_ships(SHIP_FILE,OPERATORS_FILE)    
    berth = make_berth(terminal, spots_df)
    anchor = make_anchor(port, berth)
    anchor.to_csv(f"{MODELS_PATH}/anchor.csv")

    # Find the spot_id with the minimum distance for each berth
    berth_tf = get_timeframed(berth, True, HOURS)
    anchor_tf = get_timeframed(anchor, False, HOURS)
    
    berth_tf_en = fill_tf_gaps(berth_tf, HOURS,True)
    anchor_tf_en = fill_tf_gaps(anchor_tf, HOURS,False)
    anchor_tf_en.to_csv(f"{MODELS_PATH}/anchor_tf_en.csv")
    
    # Apply the function to filter the ancor_tf_en dataframe
    anchor_tf_en['to_delete'] = anchor_tf_en.apply(lambda row: clear_or_delete(row, berth_tf_en), axis=1)

    # Clear IMO and duration where necessary
    anchor_tf_en.loc[anchor_tf_en['to_delete'], ['IMO', 'hours_spent']] = [np.nan, np.nan]

    # Remove the 'to_delete' column
    anchor_tf_en.drop(columns=['to_delete'], inplace=True)

    anchor_tf_en.to_csv(f'{MODELS_PATH}/after_clear_And_Delete.csv')########
    #filling gaps (if exit from anchor but didn't arrive to berth)
    a3 = pd.merge(anchor_tf_en, anchor_tf_en, 
                  left_on=['IMO', anchor_tf_en['timeframe_start'] + pd.to_timedelta(HOURS, unit='h')],
                  right_on=['IMO', 'timeframe_start'],
                  suffixes=('_a1', '_a2'),
                  how='left')
    # Select rows from a1 where there are no corresponding records in a2
    a3 = a3[a3['timeframe_start_a2'].isnull()]

    # Step 2: Join with berth_df_en to find final result
    final_result = pd.merge(a3, berth_tf_en,
                        left_on=['IMO', a3['timeframe_start_a1'] + pd.to_timedelta(HOURS, unit='h')],
                        right_on=['IMO', 'timeframe_start'],
                        suffixes=('_a3', '_b1'),
                        how='left')

    # Select rows from a3 where there are no corresponding records in b1
    final_result = final_result[final_result['timeframe_start_b1'].isnull()]
    a3 = a3[a3['timeframe_start_a2'].isnull()]

    # Step 2: Join with berth_tf_en to find final result
    final_result = pd.merge(a3, berth_tf_en,
                            left_on=['IMO', a3['timeframe_start_a1'] + pd.to_timedelta(HOURS, unit='h')],
                            right_on=['IMO', 'timeframe_start'],
                            suffixes=('_a3', '_b1'),
                            how='left')
    final_result = final_result[final_result['timeframe_start_b1'].isnull()]

    # Step 3: Remove rows from anchor_tf_en or leave one empty record if no other IMO
    # Remove these rows from anchor_tf_en
    rows_to_remove = final_result[['IMO', 'timeframe_start_a1']]
    rows_to_remove.columns = ['IMO', 'timeframe_start']

    anchor_tf_en = anchor_tf_en.merge(rows_to_remove, on=['IMO', 'timeframe_start'], how='left', indicator=True)
    anchor_tf_en = anchor_tf_en[anchor_tf_en['_merge'] == 'left_only'].drop(columns=['_merge'])

    # Step 4: Ensure at least one empty record for timeframes with no other IMOs
    missing_timeframes = final_result[['timeframe_start_a1']].drop_duplicates()
    missing_timeframes.columns = ['timeframe_start']
    missing_timeframes = missing_timeframes[~missing_timeframes['timeframe_start'].isin(anchor_tf_en['timeframe_start'])]

    # Create empty records for missing timeframes
    empty_records = pd.DataFrame({
        'IMO': [None] * len(missing_timeframes),
        'timeframe_start': missing_timeframes['timeframe_start'],
        'hours_spent': [None] * len(missing_timeframes)
    })
    anchor_tf_en = pd.concat([anchor_tf_en, empty_records], ignore_index=True)

    # Combine both DataFrames
    combined_df = pd.concat([anchor_tf_en.assign(dataset='anchor'), berth_tf_en.assign(dataset='berth')])
    combined_df = combined_df.sort_values(by=['IMO', 'timeframe_start']).reset_index(drop=True)
    # Find earliest positions for each segment
    earliest_positions = find_earliest_positions(combined_df, HOURS)

    # Create a DataFrame from the result
    result_df = pd.DataFrame(earliest_positions, columns=['IMO', 'timeframe_start'])

    # Generate incoming dataframe
    incoming_df = generate_incoming_dataframe(result_df, HOURS, TIMEFRAMES_PER_DAY, INCOMING_DAYS)
    
    ships_df = process_ships_dataframe(ships)
    ships_df['from_date'] = pd.to_datetime(ships_df['from_date'])
    ships_df['Date Changed'] = pd.to_datetime(ships_df['Date Changed'])    

    
    voyages_df = make_voyages()
    
    berth_tf_en_copy = berth_tf_en.copy()
    berth_tf_en_copy['spot_id'] = pd.to_numeric(berth_tf_en_copy['spot_id'], errors='coerce').astype('Int64')  # Use 'Int64' to handle NaN

    combined_df = pd.concat([berth_tf_en_copy[['timeframe_start', 'IMO', 'spot_id']],
                             anchor_tf_en[['timeframe_start', 'IMO', 'spot_id']],
                             incoming_df[['timeframe_start', 'IMO', 'spot_id']]])

    # Sort the DataFrame by IMO, spot_id, and timeframe_start
    combined_df_tf = combined_df.sort_values(by=['IMO', 'spot_id', 'timeframe_start']).reset_index(drop=True)

    # Shift the DataFrame to compare current row with previous row
    combined_df_tf['prev_timeframe_start'] = combined_df_tf.groupby(['IMO', 'spot_id'])['timeframe_start'].shift(1)

    # Calculate the difference in days between current and previous row
    combined_df_tf['time_diff'] = (combined_df_tf['timeframe_start'] - combined_df_tf['prev_timeframe_start']).dt.total_seconds() / 3600

    # Identify the start of a new sequence (where difference is more than expected or it's the first row)
    combined_df_tf['new_sequence'] = (combined_df_tf['time_diff'] > HOURS) | (combined_df_tf['time_diff'].isna())

    # Calculate cumulative sum of new_sequence to create a group id for each sequence
    combined_df_tf['sequence_id'] = combined_df_tf.groupby(['IMO', 'spot_id'])['new_sequence'].cumsum()

    # Calculate the time_spent within each sequence
    combined_df_tf['time_spent'] = combined_df_tf.groupby(['IMO', 'spot_id', 'sequence_id']).cumcount() + 1
    combined_df_tf.to_csv(f'{MODELS_PATH}/combined_tf_new.csv')
    grouped_df = combined_df_tf.groupby(['IMO', 'spot_id', 'timeframe_start'])['time_spent'].max().reset_index()
    grouped_df['spot_id'] = grouped_df['spot_id'].astype(str)
# Now convert this grouped DataFrame to a dictionary
    time_spent_lookup = grouped_df.set_index(['IMO', 'spot_id', 'timeframe_start'])['time_spent'].to_dict()
    # Drop unnecessary columns
    
    precomputed_ship_features = {}
    for ship in ships_df['IMO'].unique():
        ship_records = ships_df[ships_df['IMO'] == ship]
    
    # Loop through each record for the current ship
        for _, record in ship_records.iterrows():    
            start_date = record['from_date']
            end_date = record['Date Changed']
            current_date = start_date

        # Iterate month by month until the end_date
            while current_date <= end_date:
                month = current_date.month
                year = current_date.year

                    # Construct the key using ship, month, and year
                key = (ship, month, year)

                    # Store the precomputed ship features in the dictionary
                precomputed_ship_features[key] = [record[col] for col in sorted(ships_df.columns) if col.startswith('ops') or col.startswith('size_group')]

                    # Move to the next month
                current_date += relativedelta(months=1)    
    #first_voyage_features = next(iter(precomputed_voyage_features.values()))
    route_origin_columns = [col for col in voyages_df.columns if col.startswith('Route_Origin')]
    # Precompute voyage features
    precomputed_voyage_features = {}
    for ship in voyages_df['IMO'].unique():
        voyage_records = voyages_df[voyages_df['IMO'] == ship]
        for _, record in voyage_records.iterrows():
            start_date = record['Route_Start']
            end_date = record['Route_End']
            current_date = start_date

            # Iterate month by month until reaching the Route_End date
            while current_date <= end_date:
                month = current_date.month
                year = current_date.year

                # Construct the key using ship, month, and year
                key = (ship, month, year)

                # Store the precomputed voyage features in the dictionary
                # If the column doesn't exist in the record, use 0 as the default value
                precomputed_voyage_features[key] = [record.get(col, 0) for col in route_origin_columns]

                # Move to the next month
                current_date += relativedelta(months=1)    
    trajectories = ()
            
    if state_based:
        trajectories = extract_trajectories_state_based(berth_tf_en, anchor_tf_en, incoming_df, HOURS, terminal, state_based)
    else:
        trajectories = extract_trajectories(berth_tf_en, anchor_tf_en, incoming_df, HOURS)
    print(f"traj is {trajectories[300]}")
    '''
    time_spent_key = (ship, loc_local, timeframe_start)
    time_spent_value = local_lookup.get(time_spent_key, 0)
    
    for k in range(EXTRA_FEATURES_NUM - 1):
        if k < time_spent_value:
            features.append(1)
        else:
            features.append(0)
    '''
    print_current_time("Trajectories processed")    
    some_features = []
    if calc_features:
        tf_df = generate_df_time_window(START_DATE, END_DATE, HOURS)
        for ship in ships_df['IMO'].unique():
            for _, row in tf_df.iterrows():
                dt = row['datetime']
            # Get combined features (ship, voyage, time)
                combined_features, ship_features, voyage_features, time_features = lookup_features(ship, dt, precomputed_ship_features, precomputed_voyage_features)
            # Append to the combined_df list
                some_features.append([ship, dt] + combined_features)                     
        # Convert combined_df into a pandas DataFrame
        columns = ['ship', 'datetime'] + ['ship_feature_{}'.format(i) for i in range(len(ship_features))] \
               + ['voyage_feature_{}'.format(i) for i in range(len(voyage_features))] \
               + ['dayofweek', 'day', 'week_of_month', 'iso_week']
        some_features = pd.DataFrame(some_features, columns=columns)
    return some_features, time_spent_lookup, trajectories


def lookup_features(ship, dt, precomputed_ship_features, precomputed_voyage_features, default_value=0):
    """Returns combined features from precomputed lookups for a given ship and datetime."""
    
    # Prepare the lookup key (ship, month, year)
    key = (ship, dt.month, dt.year)
    
    # Try to get the features from the ship and voyage precomputed dictionaries
    ship_features = precomputed_ship_features.get(key, [default_value] * len(precomputed_ship_features.get(next(iter(precomputed_ship_features)), [])))
    voyage_features = precomputed_voyage_features.get(key, [default_value] * len(precomputed_voyage_features.get(next(iter(precomputed_voyage_features)), [])))
    
    # Extract the time features from the current datetime
    time_features = get_time_features(dt)
    
    # Combine ship, voyage, and time features
    return ship_features + voyage_features + time_features, ship_features, voyage_features, time_features

def policy(state, reward_function, time, lookup_predicted, feature_num, positions, features, state_based, lookup_debug = False, prediction = False):
    global berth_num
    global num_actions 
    global action_to_int
    global int_to_action 
    global scaler
    action_probs = {}
    if lookup_debug:
        print("we are in one loop of policy for state")
    state_id = 0
    for ship in state.keys():
        ship_action_probs = [0] * num_actions
        values = []
        values2 = []
        allowed_actions_for_ship = allowed_actions(state[ship], berth_num, state_based, ship)  # Get allowed actions for the ship
        # Compute the values first
        for action in allowed_actions_for_ship:
            action_dict = {ship: action}
            if prediction:
                local_features = extract_features(state, feature_num, time, lookup_predicted, positions, features, state_based, lookup_debug)
            else:
                state_key = time['timeframe_start']
                action_key = ship            
                local_features = features.get(state_key, {}).get(action_key, np.zeros(feature_num))
            
                
            if ZERO_TO_NEGATIVE:
                local_features = np.where(local_features == 0, -1, local_features)

            # Apply StandardScaler
            if  SCALE_FEATURES:
                local_features_reshaped = local_features.reshape(-1, 1)
                scaled_features = scaler.fit_transform(local_features_reshaped)
            # Reshape back to 1D
                scaled_features = scaled_features.reshape(-1)                
            else: 
                scaled_features = local_features

            if prediction:    
                local_features_noisy = scaled_features
            else: 
                noise = np.random.normal(NOISE_MEAN, NOISE_SIGMA, scaled_features.shape)
                local_features_noisy = scaled_features + noise
                
            # Continue with the reward calculation
            if USE_LOG:
                value = log_transform(np.dot(reward_function[state_id], local_features_noisy))
            #value = log_transform(np.dot(reward_function, local_features))
            else:
                value = np.dot(reward_function[state_id], local_features_noisy)
            values.append(value)
        
        if prediction: #REMOVE_WRONG_ACTIONS: 
            allowed_action_indices = [action_to_int[action] for action in allowed_actions_for_ship]
            values = np.asarray(values, dtype=np.float64)
            filtered_probs = [values[0,i] for i in allowed_action_indices]            
            #print(f"prob are {values} and filtered_probs are {filtered_probs}")
            total_prob = sum(filtered_probs)
            normalized_probs = [prob / total_prob for prob in filtered_probs]
        else:
            prob = stable_softmax(values, SOFTMAX_THRESHOLD, TEMPERATURE) #exp_values / total
            allowed_action_indices = [action_to_int[action] for action in allowed_actions_for_ship]
            filtered_probs = [prob[0,i] for i in allowed_action_indices]            
            normalized_probs = filtered_probs
        # Update ship_action_probs
        for action, prob in zip(allowed_actions_for_ship, normalized_probs):
            action_index = action_to_int[action]
            ship_action_probs[action_index] = prob        
        action_probs[ship] = ship_action_probs
        state_id += 1
    return action_probs

def predict_actions(state, time, lookup_predicted, feature_num, positions, features, reward_function, state_based, lookup_debug, prediction = True):
    global int_to_action 
    action_probs = policy(state, reward_function, time, lookup_predicted, feature_num, positions, features, state_based,  lookup_debug, prediction)
    best_actions = {}
    for ship, ship_action_probs in action_probs.items():
        print(f"for ship {ship} we have {ship_action_probs}")
        #best_action = max(ship_action_probs, key=lambda x: x[1])[0]
        best_action_index = ship_action_probs.index(max(ship_action_probs))
        best_actions[ship] = int_to_action.get(best_action_index)
    return best_actions

reward_counter = 0
def define_reward_function(reward_function, feature_num, time_spent_lookup, positions, features, states, expert_feature_expectations, state_based, next_features, terminal = 'Maher'):
    global num_actions
    global action_to_int
    global int_to_action 
    global reward_counter
    k = 0
    max_wait = 0
    if state_based:
        max_wait = load_prepared_data('max_wait', terminal, state_based) 
        max_num = max_wait
        wait_num = max_wait
        inc_num = load_prepared_data('inc_num', terminal, state_based).max() 
    
    for k in range(NUM_ITERATIONS):        
        #current_feature_expectations = np.zeros(feature_num)
        #if k == 0:
        #    print(f"curr feature num is {current_feature_expectations}")
        
        for time, state, next_state in states[:-1]: #states[:-1]:
            # Initialize feature expectations for this state
            current_feature_expectations = [] #np.zeros((num_actions, feature_num))
            observed_feature_expectations = [] #np.zeros((num_actions, feature_num))
            #action_probs = predict_actions(state, time, time_spent_lookup, feature_num, positions, features, reward_function, state_based, lookup_debug = False, prediction = False)
            action_probs = policy(state, reward_function, time, time_spent_lookup, feature_num, positions, features, state_based)
            next_state_key = time['timeframe_start'] + pd.Timedelta(hours=HOURS)
            reward_counter = 0
            for ship, ship_action_probs in action_probs.items():
                allowed_actions_for_ship = allowed_actions(state[ship], berth_num, state_based, ship)
                
                for action in allowed_actions_for_ship:
                    action_idx = action_to_int[action]
                    if USE_LOG:
                        log_prob = ship_action_probs[action_idx]
                    else:
                        log_prob = np.log(ship_action_probs[action_idx])  
                    #if ship == 'IMO9484376':
                    #    print(f" we have ship {ship} and  action {[action]} and {ship_action_probs}")
                    #update next_stete by changing this ship position only somehow
                    expected_next_state = next_state.copy()
                    if action != 'leave_system':
                        ns = get_next_state(action, state[ship], expected_next_state, state_based, max_wait)
                        #if ship == 'IMO9484376':
                        #    print(f" we have action {[action]} and ns {ns} AND state {state[ship]}")
                        expected_next_state[ship] = ns
                    else:
                        if ship in expected_next_state:
                            del expected_next_state[ship]
                    time_spent_lookup_next_state = time_spent_lookup.copy()
                    for loc in list(time_spent_lookup_next_state.keys()):
                        if loc[2] == next_state_key and loc[0] == ship:
                            del time_spent_lookup_next_state[loc]
                    if action != 'leave_system':
                        loc_local = expected_next_state[ship]
                        if state_based:
                            loc_local = str(loc_local).rsplit('_', 1)[0]
                        if expected_next_state[ship] == state.get(ship):
                            current_position_key = (ship, loc_local, next_state_key)
                            prev_datetime = time['timeframe_start']
                            prev_position_key = (ship, state.get(ship), prev_datetime)
                            if prev_position_key in time_spent_lookup_next_state:
                                time_spent_lookup_next_state[current_position_key] = time_spent_lookup_next_state[prev_position_key] + 1
                            else:
                                time_spent_lookup_next_state[current_position_key] = 1
                        else:
                            # Add new position for ships that appeared at a new place
                            new_position_key = ( ship, loc_local, next_state_key)
                            time_spent_lookup_next_state[new_position_key] = 1 #0????
#                key = (ship, loc_local, next_state_key)

                #update time_spent_lookup for this position
                    '''
            # Update lookup_predicted
            for ship, new_position in new_state_expected.items():
                # Delete for new time all rows
                loc_local = new_position
                if state_based:
                    loc_local = str(new_position).rsplit('_', 1)[0]


                    new_time_key = (next_state_key, ship)
                for loc in list(time_spent_lookup_next_state.keys()):
                    if loc[2] == new_time_key[0] and loc[0] == new_time_key[1]:
                        del time_spent_lookup_next_state[loc]
                prev_position_key =()
                # Increment for ships staying at the same place
                if loc_local == state.get(ship):
                    current_position_key = (ship, loc_local, next_state_key)
                    prev_datetime = time['timeframe_start']
                    prev_position_key = (ship, new_position, prev_datetime)
                    if prev_position_key in time_spent_lookup_next_state:
                        time_spent_lookup_next_state[current_position_key] = time_spent_lookup_next_state[prev_position_key] + 1
                    else:
                        time_spent_lookup_next_state[current_position_key] = 1
                else:
                    # Add new position for ships that appeared at a new place
                    new_position_key = ( ship, loc_local, next_state_key)
                    time_spent_lookup_next_state[new_position_key] = 1 #0????
                key = (ship, loc_local, next_state_key)
                    '''
            # Apply probabilities to the feature vector for the current state
                    current_feature_expectations = extract_features(expected_next_state, feature_num, time, time_spent_lookup_next_state, positions, next_features, state_based, False)
                    observed_feature_expectations = features.get(next_state_key, {}).get(ship, np.zeros(feature_num))
                    #print(f"new_state_expected is {expected_next_state}")    
                    #print(f"next_state actual is {next_state}")    

                # Compute the gradient between current and observed feature expectations
                    gradient = current_feature_expectations - observed_feature_expectations            
                    #if ship == 'IMO9827334':
                    #    print(f"new_state_expected is {expected_next_state}")    
                    #    print(f"next_state actual is {next_state}")    
                        
                    #    np.set_printoptions(threshold=np.inf)
                    #    print(f"GRADIENT for iteration {k}")
                    #    print(LEARNING_RATE * (gradient + L1_REG * np.sign(reward_function[reward_counter][action_idx]) + L2_REG * reward_function[reward_counter][action_idx]))
                    #    np.set_printoptions(threshold=1000)
                    reward_function[reward_counter][action_idx] -= LEARNING_RATE * (log_prob * gradient + L1_REG * np.sign(reward_function[reward_counter][action_idx]) + L2_REG * reward_function[reward_counter][action_idx])
                reward_counter += 1
            
            '''
            for ship, ship_action_probs in action_probs.items():
                state_key = time['timeframe_start']
                action_key = ship

                # Fetch the feature vector for this state and action; shape should be (2541,)
                local_features = features.get(state_key, {}).get(action_key, np.zeros(feature_num))
                #print(f"prob is {ship_action_probs}")

                probabilities = np.array(ship_action_probs).reshape(-1, 1)  # Corresponding probabilities
                #probabilities
                # Create a matrix for probabilities with shape (num_actions, 1)
                #probabilities_matrix = np.zeros((num_actions, num_actions))
                #probabilities_matrix[:, np.arange(num_actions)] = probabilities

                # Reshape local_features to match (num_actions, feature_num)
                local_features_matrix = np.tile(local_features, (num_actions, 1))  # Shape (num_actions, feature_num)

                # Compute the weighted features
                #feature_updates = local_features_matrix * probabilities_matrix.T  # Shape (num_actions, feature_num)
                trt = False
                if np.any(np.isnan(local_features)):
                    print(f"nan happened for local features  {state_key} and {action_key}")
                    trt = True
                if np.any(np.isnan(probabilities)):
                    print(f"nan happened for probabilities  {state_key} and {action_key}")  
                    policy(state, reward_function, time, time_spent_lookup, feature_num, positions, features, state_based, True)
                    trt = True
                if np.any(np.isnan(local_features_matrix)):
                    print(f"nan happened for local features matrix {state_key} and {action_key}")
                    trt = True
                if trt:
                    sys.exit(0)
                feature_updates = local_features_matrix * probabilities
                # Update current_feature_expectations
                current_feature_expectations += feature_updates     
            next_state_key = time['timeframe_start'] + pd.Timedelta(hours=HOURS)
            #print(f"next_step_key is {next_state_key} and next_state is {next_state}")
            for ship in next_state:                
                action_key = ship
                #print(f"action_key is {action_key}")
                # Fetch the feature vector for this state and action; shape should be (2541,)
                loc_features_2 = features.get(next_state_key, {}).get(action_key, np.zeros(feature_num))
                observed_feature_expectations += loc_features_2
                
                //OLD
        for time, state in states:
            action_probs = policy(state, reward_function, time, time_spent_lookup, feature_num, positions, features, state_based)
            for ship, ship_action_probs in action_probs.items():
                for action, prob in ship_action_probs:
                    state_key = time['timeframe_start']
                    action_key = tuple(ship)            
                    #state_key = (time, state, ship)
                    #action_key = action
                    local_features = features.get(state_key, {}).get(action_key, np.zeros(feature_num))
                    action_number = action_to_int.get(action)
                    print(f"Shape of current_feature_expectations[action_number]: {current_feature_expectations[action_number].shape}")  # Should be (2541,)
                    print(f"Shape of local_features: {local_features.shape}")  # Should be (2541,)
                    print(f"Type and shape of prob: {type(prob)}, {np.shape(prob)}")  
            # Accumulate feature expectations for each action separately
                    current_feature_expectations[action_number] += local_features * prob            
                
        for time, state in states:
            action_probs = policy(state, reward_function, time, time_spent_lookup, feature_num, positions, features, state_based)
            for ship, ship_action_probs in action_probs.items():
                state_key = time['timeframe_start']
                action_key = tuple(ship)
                #local_features = extract_features(state, feature_num, time, time_spent_lookup, positions, features, state_based)
                local_features = features.get(state_key, {}).get(action_key, np.zeros(feature_num))
                current_feature_expectations[action] += local_features * prob
                #for action, prob in ship_action_probs:
                    #action_dict = {ship: action}
        '''
                    #features.get(state_key, {}).get(action_key, np.zeros(feature_num))     !!!!!!!!!
                 #   current_feature_expectations += local_features * prob

            #current_feature_expectations /= len(states)
            #observed_feature_expectations /= len(states)
        print_current_time(f"iteration {k}")
        if k == 0:
            print_current_time(f"iteration {k}")
        if k % 10 == 0:
            print_current_time(f"{k} iterations passed")
            save_prepared_data(reward_function, f'reward_function_{k}', terminal, state_based)
    return reward_function

def predict_steps_ahead(initial_state, initial_time, steps, tsl, feature_num, positions, features, reward_function, state_based, terminal):
    global max_wait
    global wait_num
    global inc_num
    predictions = []
    state = initial_state
    time = initial_time
    lookup_predicted = tsl.copy()  # Assuming time_spent_lookup is already defined
    debug = False
    lookup_debug = False #!!!!!
    max_wait = 0
    if state_based:
        max_wait = load_prepared_data('max_wait', terminal, state_based) 
        max_num = max_wait
        wait_num = max_wait
        inc_num = load_prepared_data('inc_num', terminal, state_based).max() 
    if state_based:
        xgboost_model = load_prepared_data('xgboost_model', terminal, state_based) # no irl prefix  
    for step in range(steps):
        if debug:
            print(step, state)
        if step > 0:
            lookup_debug == False
        action = predict_actions(state, time, lookup_predicted, feature_num, positions, features, reward_function, state_based, lookup_debug)
        #action2 = get_xgboost_prediction(xgboost_model, state, feature_num, time, lookup_predicted, positions, features, state_based, lookup_debug)
        if True: 
            print('XGBoost')
            #print(action2)
            print('IRL')
            print(action)
        if debug:
            print(action)
        new_state = action_to_state(action, state, state_based, max_wait)
        #new_state2 = action_to_state(action2, state, state_based, max_wait)
        if debug:
            print(new_state)
        predictions.append((
            time,
            state,
            action,
            new_state
        ))

        # Update lookup_predicted
        for ship, new_position in new_state.items():
            # Delete for new time all rows
            loc_local = new_position
            if state_based:
                loc_local = str(new_position).rsplit('_', 1)[0]
            

            new_time_key = (time['timeframe_start'], ship)
            for loc in list(lookup_predicted.keys()):
                if loc[2] == new_time_key[0] and loc[0] == new_time_key[1]:
                    del lookup_predicted[loc]
            prev_position_key =()
            # Increment for ships staying at the same place
            if loc_local == state.get(ship):
                current_position_key = (ship, loc_local, time['timeframe_start'])
                prev_datetime = time['timeframe_start'] - timedelta(hours=HOURS)
                prev_position_key = (ship, loc_local, prev_datetime)
                if prev_position_key in lookup_predicted:
                    lookup_predicted[current_position_key] = lookup_predicted[prev_position_key] + 1
                else:
                    lookup_predicted[current_position_key] = 1
            else:
                # Add new position for ships that appeared at a new place
                new_position_key = ( ship, loc_local, time['timeframe_start'])
                lookup_predicted[new_position_key] = 1 #0????
            key = (ship, loc_local, time['timeframe_start'])
            if debug:
                print(f"for {key} predicted is {lookup_predicted[key]}")
            if prev_position_key in lookup_predicted:
                if debug:
                    print(f"for {prev_position_key} prev_position_key is {lookup_predicted[prev_position_key]}")
            if key in tsl:
                if debug:
                    print(f"for {key} actual is {tsl[key]}")
            else:
                if debug:
                    print(f"for {key} actual is not exists")
                if ship in ('IMO9612870', 'IMO9290103', 'IMO9395020', 'IMO9894961'):
                    if debug:
                        print(ship)
                        print(extract_remaining_parts(tsl, ship))
                        print(list_values_for_key_part(tsl, ship))
                    
                
        # Increment the time
        rounded_datetime = time['timeframe_start']
        rounded_datetime += timedelta(hours=HOURS)  # or appropriate time increment

        # Update the time dictionary
        time = {
            'timeframe_start': rounded_datetime,
            'day_of_week': rounded_datetime.weekday(),
            'day_of_month': rounded_datetime.day,
            'week_of_month': get_week_of_month(rounded_datetime),
            'week_of_year': rounded_datetime.isocalendar()[1]
        }
        state = new_state
    return predictions, lookup_predicted

def run_xgboost(trajectories, state_based, feature_num, max_ships, positions, features):
    global action_to_int
    global int_to_action    
    global num_actions
    #one_hot_move = action_to_one_hot('move', action_to_int, num_actions)
    #let's try to train the model
    x_len = max_ships * feature_num
    y_len = num_actions * max_ships
    
    if state_based:
        y_len = num_actions * len(positions) # need to ork on this as 0 could be padded in the middle also
        x_len = feature_num * len(positions)

    X = np.empty((0, x_len))
    y = np.empty((0, y_len))
        
    for time, state, actions, next_state in trajectories[:-1]:
        state_key = time['timeframe_start']
        current_x = np.empty(0)
        current_y = np.empty(0)
        if state_based:
            #current_y = np.zeros((len(positions)))
            ship_key = None
            for pos in positions:
                ship_key = None
                pos_key = None
                for ship, pos_of_ship in state.items():
                    if ship != None:
                        ship_key = ship
                        pos_key = pos_of_ship
                        break
                if pos_of_ship == pos:    
                    current_y = np.hstack([current_y, action_to_one_hot(actions[ship], action_to_int, num_actions)])
                    current_x = np.hstack([current_x, features.get(state_key, {}).get(ship_key, np.zeros(feature_num))])
                else:
                    current_y = np.hstack([current_y, np.zeros(( num_actions), dtype=int)])
                    current_x = np.hstack([current_x, np.zeros((feature_num), dtype=int)])
            
        else:
            for action in actions.values():
                if action == 'stay_at_berth_incoming_5':
                    print(f"actions are {actions} for time {time} and state {state}")
                else:                
                    current_y = np.hstack([current_y, action_to_one_hot(action, action_to_int, num_actions)])

            len_curr_y = len(current_y)
            if y_len > len_curr_y:
                # For each missing step, append a zero-filled vector
                #current_y +=  np.zeroes(y_len - len_curr_y, dtype=int)
                current_y = np.hstack([current_y, np.zeros((y_len - len_curr_y), dtype=int)])
            state_key = None
            for ship, pos_of_ship in state.items():
                if ship != None:
                    ship_key = ship
                    current_x =  features.get(state_key, {}).get(ship, np.zeros(feature_num))                 
            
            len_curr_x = len(current_x)
            if x_len > len_curr_x:
                current_x = np.hstack([current_x, np.zeros((x_len - len_curr_x), dtype=int)])
        X = np.vstack([X, current_x])
        y = np.vstack([y, current_y])

    print("Shape of X:", X.shape)
    print("Shape of y:", y.shape)
    model = LinearRegression()

    # Train the model
    model.fit(X, y)    
    return model

def get_xgboost_prediction(model, state, feature_num, time, lookup_predicted, positions, features, state_based, lookup_debug):
    global num_actions
    #print("I'm here")
    global int_to_action
    #print(f"int_to_action is {int_to_action}")
    global loc_list
    #print(f"loc_list is {loc_list}")
    X_test = extract_features(state, feature_num, time, lookup_predicted, positions, features, state_based, lookup_debug)
    #print(f"x_test is {X_test}")
    y_pred = model.predict(X_test.reshape(1, -1)) 
    
    
    num = len(y_pred) // num_actions  # Ensure y_pred length is divisible by num_actions

    # Step 1: Split y_pred into 'num' pieces and find the most probable action in each piece
    one_hot_predictions = []
    actions = []
    for i in range(num):
        # Get the slice corresponding to this action's prediction (e.g., for 'move' and 'stay')
        pred_piece = y_pred[i*num_actions : (i+1)*num_actions]

        # Step 2: Find the index of the highest probability
        max_index = np.argmax(pred_piece)

        # Step 3: Create a one-hot encoded vector (1 at max_index, rest 0)
        one_hot_vector = np.zeros(num_actions, dtype=int)
        one_hot_vector[max_index] = 1

        # Store the one-hot encoded vector
        #one_hot_predictions.append(one_hot_vector)
        get_actions = one_hot_to_action(one_hot_vector, int_to_action)
        # Step 4: Map back to the action using max_index
        predicted_action = actions[max_index]
        print(f"Piece {i+1}: Predicted action is '{predicted_action}' with one-hot vector {one_hot_vector}")
        actions.append(predicted_action)
        if state_based:
            for ship, pos_a in state.items():
                if pos_a == loc_list[i]:
                    ship_actions[ship] = predicted_action
                    break
        else:
            items_list = list(state.items())
            ship, value = items_list[index]
            ship_actions[ship] = predicted_action
    return ship_actions
    
def run(terminal = 'NY_Maher', port = 'NY', data_preparation_flag = True, reward_function_calculation_flag = True, prediction_flag = True, state_based = True):
    global possible_actions
    global ships_features
    global route_origin_length
    global berth_num
    global wait_num 
    global inc_num 
    global loc_list
    global max_list
    global action_to_int
# Create number -> action mapping (reverse mapping)
    global int_to_action 

    print_current_time("Start time")    
    spots_df = make_spots(terminal)
    if data_preparation_flag:
        print_current_time("Data Preparation Started")    
        #trajectories, combined_df_tf, time_spent_lookup, precomputed_ship_features, precomputed_voyage_features = data_preparation(terminal, port, spots_df,state_based)
        calc_features = False
        features, time_spent_lookup, trajectories = feature_lookup_preparation(terminal, port, spots_df,state_based, calc_features)
        print_current_time("After preparing data")    
        # Get unique spot_id values

        save_prepared_data(trajectories, 'trajectories_new', terminal, state_based)
        save_prepared_data(time_spent_lookup, 'time_spent_lookup_new', terminal, state_based) # no irl prefix
        if calc_features:
            save_prepared_data(features, 'features', 'EC', False)
        print_current_time("After dumping data to disk")    
    else:
        trajectories = load_prepared_data('trajectories_new', terminal, state_based)
        #combined_df_tf = load_prepared_data('combined_df_tf_new', terminal, state_based) # no irl prefix
        time_spent_lookup = load_prepared_data('time_spent_lookup_new', terminal, state_based) # no irl prefix
        #features = load_prepared_data('features', 'EC', False)
        if state_based:
            max_wait = load_prepared_data('max_wait', terminal, state_based) 
            max_num = max_wait
            wait_num = max_wait
            inc_num = load_prepared_data('inc_num', terminal, state_based).max()         
    features = load_prepared_data('features', 'EC', False)   
    features['datetime'] = pd.to_datetime(features['datetime'], errors='coerce')
    max_ships = max(len(state) for _, state, _, _ in trajectories)
    berth_num = spots_df['spot_id'].dropna().unique()
    print(f"berth_num is {berth_num}")
    actions = define_actions(berth_num)
    action_to_int = {action: i for i, action in enumerate(actions)}
# Create number -> action mapping (reverse mapping)
    int_to_action = {i: action for i, action in enumerate(actions)}
    print(f"action_to_int {action_to_int}")
    print(f"int_to_action {int_to_action}")
    possible_actions = actions
    positions = berth_num + 1 + INCOMING_DAYS * TIMEFRAMES_PER_DAY #combined_df_tf['spot_id'].dropna().unique()
    print(positions)
    loc_list = None
    max_list = None 
    if state_based:
        loc_list = get_loc_list(define_states(berth_num))
        max_list = len(loc_list)
    ships_features = len(features.columns) - 2
    feature_num = max_ships * (#len(possible_actions) + 
                                len(positions) + ships_features + EXTRA_FEATURES_NUM)
    features.set_index(['datetime', 'ship'], inplace=True)
    states_num = max_ships
    if state_based:
        feature_num = (#len(possible_actions) + 
            ships_features + EXTRA_FEATURES_NUM) * max_list
        states_num = max_list
    print(f"feature_num is {feature_num} and max_ships {max_ships} and ships_features {ships_features}")
    global num_actions
    num_actions = len(actions)
    
    reward_function = np.zeros((states_num, num_actions, feature_num))
    print(f"starting precompute_Features")
    precomputed_features, trajectories = precompute_features(trajectories, feature_num, time_spent_lookup, positions, features, state_based)
    #np.set_printoptions(threshold=np.inf)
    print('-----precomputed_features-')
    #print(precomputed_features)
    #np.set_printoptions(threshold=1000)

    print("Sucessfully loaded data")    
    if reward_function_calculation_flag:
        print("Start of Reward Function calculation")
        states = [(trajectory[0], state, next_state) for trajectory in trajectories for state in [trajectory[1]] for next_state in [trajectory[3]]]
        expert_feature_expectations = compute_expert_feature_expectations(trajectories, feature_num, time_spent_lookup, positions, features, state_based)
        reward_function = define_reward_function(reward_function, feature_num, time_spent_lookup, positions, precomputed_features, states, expert_feature_expectations, state_based, features, terminal)
        save_prepared_data(reward_function, 'reward_function', terminal, state_based)
        print("End of Reward Function calculation")
    #xgBoost = True
    xgBoost = False
    if xgBoost:
        print("Start of XGBoost Training")
        xgboost_model = run_xgboost(trajectories, state_based, feature_num, max_ships, positions, precomputed_features)
        save_prepared_data(xgboost_model, 'xgboost_model', terminal, state_based) # no irl prefix  
    if prediction_flag:
        print("Start of predicting")
        if not reward_function_calculation_flag:
            reward_function = load_prepared_data('reward_function', terminal, state_based)
        #aaaa = np.get_printoptions()
        #print(aaaa)
        #np.set_printoptions(threshold=np.inf)
        #print(reward_function)
        
        #np.set_printoptions(threshold=1000)
        #print('---------------------')
        start_traj_list = [300, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 12500]
        number_of_steps = 10
        time_spent_lookup_actual = time_spent_lookup.copy()
        dfs = []
        accuracy = []
        occupation = []
        #start_traj_list = [300]
        for start_traj in start_traj_list:
            initial_state = trajectories[start_traj-1][1]
            initial_time = trajectories[start_traj-1][0]
            print(f"for {start_traj} initial_state is  {initial_state}")
            # looping reward functions
            reward_iterations = 1
            if True:
                reward_iterations = NUM_ITERATIONS // 10 + 1
            for rew in range(reward_iterations):
                if reward_iterations > 1:
                    reward_function = load_prepared_data(f'reward_function_{rew*10}', terminal, state_based)
                else:
                    reward_function = load_prepared_data(f'reward_function', terminal, state_based)
                print(f"----------------------REWARD FUNCTION AFTER {rew} ITERATIONS----------------")
                #np.set_printoptions(threshold=np.inf)
                #print(reward_function)
        
                #np.set_printoptions(threshold=1000)

                for n in range(number_of_steps):
                    print(f"---------------{n}-----------------")
                    predicted_trajectories, time_spent_lookup_predicted = predict_steps_ahead(initial_state, initial_time, n+1, time_spent_lookup.copy(), feature_num, positions, features, reward_function,state_based, terminal)

                    actual_trajectories = trajectories[start_traj-1:start_traj + n] #removed (-1) as I need at least 1 trajectory

                    #print_traj(actual_trajectories, predicted_trajectories)        
                    comparison_results = compare_time_spent(actual_trajectories, predicted_trajectories, time_spent_lookup_actual, time_spent_lookup_predicted, state_based)
                    occupation_results, occupation_df = compare_occupation(actual_trajectories, predicted_trajectories, state_based)
                    occupation_df['steps'] = n + 1
                    occupation_df['period_start'] = start_traj
                    occupation_df['reward_num'] = rew
                    occupation.append(occupation_df)
                    print("occupation_results")
                    print(occupation_results)
                    print("comparison_results")
                    df_comparison_results = pd.DataFrame(comparison_results)
                    df_comparison_results['steps'] = n + 1
                    df_comparison_results['period_start'] = start_traj
                    df_comparison_results['reward_num'] = rew
                    # Append the DataFrame to the list
                    dfs.append(df_comparison_results)
                    # Display the DataFrame
                    print(df_comparison_results)

                    print("prediction_accuracy")
                    precise, general = calculate_accuracy(actual_trajectories, predicted_trajectories, state_based)
                    df_acc = pd.DataFrame({
                        'precise': [precise],  # Repeat scalar value
                        'general': [general],  # Repeat scalar value
                        'steps': [n + 1],          # Repeat scalar value
                        'period_start': [start_traj],  # Repeat scalar value
                        'reward_num': [rew]
                    })
                    accuracy.append(df_acc)

            df_combined = pd.concat(dfs, ignore_index=True)
            df_combined.to_csv(f"{MODELS_PATH}/stay_prediction_{terminal}_{state_based}_1.csv")
            df_accuracy = pd.concat(accuracy, ignore_index=True)
            df_accuracy.to_csv(f"{MODELS_PATH}/accuracy_{terminal}_{state_based}_1.csv")
            df_occupation = pd.concat(occupation, ignore_index=True)
            df_occupation.to_csv(f"{MODELS_PATH}/occupation_{terminal}_{state_based}_1.csv")

            print("end of predicting")
def print_consts():
    print(f"DATA_PATH={DATA_PATH}")
    print(f"MODELS_PATH={MODELS_PATH}")
    print(f"BERTH_FILE={BERTH_FILE}")
    print(f"ANCHOR_FILE={ANCHOR_FILE}")
    print(f"SHIP_FILE={SHIP_FILE}")
    print(f"OPERATORS_FILE={OPERATORS_FILE}")
    print(f"VOYAGES_FILE={VOYAGES_FILE}")
    print(f"MIN_BERTH_TIME={MIN_BERTH_TIME}")
    print(f"MIN_ANCHOR_TIME={MIN_ANCHOR_TIME}")
    print(f"HOURS={HOURS}")
    print(f"TIMEFRAMES_PER_DAY={TIMEFRAMES_PER_DAY}")
    print(f"INCOMING_DAYS={INCOMING_DAYS}")
    print(f"EXTRA_FEATURES_NUM={EXTRA_FEATURES_NUM}")
    print(f"TIME_FEATURES={TIME_FEATURES}")
    print(f"NUM_ITERATIONS={NUM_ITERATIONS}")
    print(f"LEARNING_RATE={LEARNING_RATE}")
    print(f"START_DATE={START_DATE}")
    print(f"END_DATE={END_DATE}")
    print(f"L1_REG={L1_REG}")
    print(f"L2_REG={L2_REG}")
    print(f"SOFTMAX_THRESHOLD={SOFTMAX_THRESHOLD}")
    print(f"TEMPERATURE={TEMPERATURE}")
    print(f"STAY_MULTIPLICATOR={STAY_MULTIPLICATOR}")
    print(f"NOISE_MEAN={NOISE_MEAN}")
    print(f"NOISE_SIGMA={NOISE_SIGMA}")
    print(f"USE_LOG={USE_LOG}")

    
def run_terminal(kk):
    k = kk
    k = 1

    data_preparation_flag = False
    reward_function_calculation_flag = True
    prediction_flag = True
    print(f"data_preparation_flag={data_preparation_flag}")
    print(f"reward_function_calculation_flag={reward_function_calculation_flag}")
    print(f"prediction_flag={prediction_flag}")
    print_consts()
    if k==1:
        print('Skip NY_Maher True')
        run(terminal = 'NY_Maher', port = 'NY', data_preparation_flag = data_preparation_flag, reward_function_calculation_flag = reward_function_calculation_flag, prediction_flag = prediction_flag, state_based = True)
    elif k == 2:
        print('Skip NY_Maher False')
        run(terminal = 'NY_Maher', port = 'NY', data_preparation_flag = data_preparation_flag, reward_function_calculation_flag = reward_function_calculation_flag, prediction_flag = prediction_flag, state_based = False)
    elif k == 3:
        print('Skip Savannah True')
        run(terminal = 'Savanna', port = 'Savanna', data_preparation_flag = data_preparation_flag, reward_function_calculation_flag = reward_function_calculation_flag, prediction_flag = prediction_flag, state_based = True)
    elif k == 4:
        print('Skip Savannah False')
        run(terminal = 'Savanna', port = 'Savanna', data_preparation_flag = data_preparation_flag, reward_function_calculation_flag = reward_function_calculation_flag, prediction_flag = prediction_flag, state_based = False)
    elif k == 5:
        print('Skip NY_APM True')
        run(terminal = 'NY_APM', port = 'NY', data_preparation_flag = data_preparation_flag, reward_function_calculation_flag = reward_function_calculation_flag, prediction_flag = prediction_flag, state_based = True)
    else:
        print('Skip NY_APM False')
        run(terminal = 'NY_APM', port = 'NY', data_preparation_flag = data_preparation_flag, reward_function_calculation_flag = reward_function_calculation_flag, prediction_flag = prediction_flag, state_based = False)

#run_terminal(2)

def print_reward(terminal, state_based):
    reward_function = load_prepared_data('reward_function', terminal, state_based)
    print(f"for terminal {terminal} and state_based {state_based} the reward function has {len(reward_function)}")
#print_reward('NY_APM', True)
#print_reward('NY_APM', False)

# print_reward('NY_Maher', True)
# print_reward('NY_Maher', False)
