from IRLconstants import DATA_PATH, MODELS_PATH, BERTH_FILE, ANCHOR_FILE, SHIP_FILE, OPERATORS_FILE, VOYAGES_FILE, MIN_BERTH_TIME, MIN_ANCHOR_TIME, HOURS, TIMEFRAMES_PER_DAY, INCOMING_DAYS, EXTRA_FEATURES_NUM
from Utils import print_current_time, round_down_to_h_window, get_week_of_month, str_to_float_tuple, calculate_distances

def define_actions(berth_num):
    actions = []
    actions.append('leave_system')
    for days in range(TIMEFRAMES_PER_DAY * INCOMING_DAYS -1):
        actions.append(f'move_closer_to_incoming_{days+1}')
    # Assigning to berth spots
    for berth in berth_num:
        actions.append(f'move_to_berth_{int(berth)}')
        actions.append(f'stay_at_berth_{int(berth)}')
    # Moving to waiting zone
    actions.append('move_to_wait_zone')
    actions.append('stay_at_wait_zone')    
    return actions

def define_states(berth_num):
    actions = []
    for days in range(TIMEFRAMES_PER_DAY * INCOMING_DAYS -1):
        actions.append(f'incoming_{days+1}')
    # Assigning to berth spots
    for berth in berth_num:
        actions.append(f'{int(berth)}')
    # Moving to waiting zone
    actions.append('0.0')
    return actions

def allowed_actions(state, berth_num, is_state = False, ship = ""): 
        
    actions = []
    state_str = str(state) 
    if is_state:
        state_str = state_str.rsplit('_', 1)[0]
    max_berth = int(max(berth_num)) 
    if state_str in [f'{i}' for i in range(1, max_berth + 1)]:
        actions = ['leave_system']
        actions.append(f'stay_at_berth_{int(state_str.split("_")[-1])}')
    elif state_str == '0.0' or state_str == '0':
        actions = ['stay_at_wait_zone']
        for berth in range(max_berth):
            actions.append(f'move_to_berth_{berth+1}')
    elif state_str.startswith('incoming_'):
        kk = int(state_str.split('_')[-1])
        if kk == 1:
            actions = ['move_to_wait_zone']
            for berth in range(max_berth):
                actions.append(f'move_to_berth_{berth+1}')
        else:
            actions.append(f'move_closer_to_incoming_{kk-1}')
    else:
        print(f'UNKNOWN ACTION FOR STATE {state} and {state_str} and {ship}')    
    return actions

def get_next_empty_zone(new_states, max_wait):
    # Collect all keys that start with '0.0_'
    keys_with_prefix = {key for key in new_states.keys() if key.startswith('0.0_')}
    
    # Search for the first missing '0.0_i' key
    for i in range(1, max_wait + 1):
        candidate_key = f'0.0_{i}'
        if candidate_key not in keys_with_prefix:
            return candidate_key
    print("all wait zzones have taken!!!")
    # If all up to max_wait are taken, return the next available one
    return f'0.0_{max_wait + 1}'

def action_to_state(action_dict, state, state_based, max_wait):
    # Initialize a dictionary to store the new states for each ship
    new_states = state.copy()
    
    # Process actions in the specified order:
    action_priority = ['leave_system', 'stay_at_berth', 'move_to_berth', 'stay_at_wait_zone', 'move_to_wait_zone', 'move_closer_to_incoming']

    # Sort actions according to the priority list
    sorted_actions = sorted(
        action_dict.items(), 
        key=lambda item: action_priority.index(
            item[1] if item[1] in ['leave_system', 'stay_at_wait_zone', 'move_to_wait_zone'] 
            else '_'.join(item[1].split('_')[:-1])
        )
    )
    for ship, action in sorted_actions:
        # If state-based conversion is enabled
        if state_based:
            if action == 'leave_system':
                # Clear berth state if ship is leaving the system
                '''
                for berth, ship_at_berth in new_states.items():
                    if ship_at_berth == ship:
                        new_states[berth] = None
                        break
                '''
                del new_states[ship]
            elif action.startswith('move_to_berth_'):
                berth = f"{action.split('_')[-1]}_1"
                '''
                if berth in new_states and new_states[berth] is not None:
                    raise ValueError(f"Berth {berth} is already occupied by another ship!")
                 # If ship is moving from a waiting area, clear the waiting area state
                
                for state, ship_at_state in new_states.items():
                    if ship_at_state == ship:
                        new_states[state] = None
                        break
                new_states[berth] = ship                
                '''
                for ship_at_state, state in new_states.items():
                    if state == berth and ship_at_state != ship:
                        #raise ValueError(f"Berth {berth} is already occupied by another ship!")
                        print(f"Berth {berth} is already occupied by another ship!")
                new_states[ship] = berth
            #elif action == 'stay_at_wait_zone':
                #do nothing
            elif action == 'move_to_wait_zone':
                # Find an empty waiting zone
                #empty_zone = next((f'0.0_{i}' for i in range(1, max_wait + 1) if new_states.get(f'0.0_{i}') is None), None)
                #if empty_zone is None:
                #    raise ValueError("No empty waiting zones available!")
                #new_states[empty_zone] = ship
                new_states[ship] = get_next_empty_zone(new_states, max_wait) 
            elif action.startswith('move_closer_to_incoming_'):
                incoming_step = int(action.split('_')[-1])
                state_num = int(new_states[ship].split('_')[-1])
                new_st = f'incoming_{incoming_step}_{state_num}'
                new_states[ship] = new_st
                '''
                for state, ship_at_state in new_states.items():
                    if ship_at_state == ship:
                        state_num = int(ship_at_state.split('_')[-1])
                        current_st = f'incoming_{incoming_step + 1}_{state_num}'
                        new_st = f'incoming_{incoming_step}_{state_num}'
                        new_states[current_st] = None
                        new_states[new_st] = ship
                        break        
                '''        
        else:  # Old conversion logic
            if action.startswith('move_closer_to_incoming_'):
                days = int(action.split('_')[-1])
                new_states[ship] = f'incoming_{days}'
            elif action.startswith('move_to_berth_'):
                berth = action.split('_')[-1]
                new_states[ship] = berth
            elif action.startswith('stay_at_berth_'):
                berth = action.split('_')[-1]
                new_states[ship] = berth
            elif action == 'move_to_wait_zone':
                new_states[ship] = '0.0'
            elif action == 'stay_at_wait_zone':
                new_states[ship] = '0.0'
            elif action == 'leave_system':
                # Ship leaves the system, do not add to new_states
                if ship in new_states:
                    del new_states[ship]
            else:
                raise ValueError(f'Unknown action: {action}')
    
    return new_states

def get_next_state(action, ship_state,  state, state_based, max_wait):
    # Initialize a dictionary to store the new states for each ship
    #new_states = state.copy()
    
    # Process actions in the specified order:
    action_priority = ['leave_system', 'stay_at_berth', 'move_to_berth', 'stay_at_wait_zone', 'move_to_wait_zone', 'move_closer_to_incoming']
    if True: # lazy to change ident
        if state_based:
            if action == 'leave_system':
                return "left system"
            elif action.startswith('move_to_berth_'):
                return f"{action.split('_')[-1]}_1"
            elif action == 'move_to_wait_zone':
                return get_next_empty_zone(state, max_wait) #!!!!!!
            elif action.startswith('move_closer_to_incoming_'):
                incoming_step = int(action.split('_')[-1])
                state_num = int(ship_state.split('_')[-1])
                return f'incoming_{incoming_step}_{state_num}'
            else:
                return ship_state
        else:  # Old conversion logic
            if action.startswith('move_closer_to_incoming_'):
                days = int(action.split('_')[-1])
                return f'incoming_{days}'
            elif action.startswith('move_to_berth_'):
                return action.split('_')[-1]                
            elif action.startswith('stay_at_berth_'):
                return action.split('_')[-1]
            elif action == 'move_to_wait_zone':
                return '0.0'
            elif action == 'stay_at_wait_zone':
                return '0.0'
            elif action == 'leave_system':
                # Ship leaves the system, do not add to new_states
                return "left system"
            else:
                raise ValueError(f'Unknown action: {action}')
    
    return new_states

