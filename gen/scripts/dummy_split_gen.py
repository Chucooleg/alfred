import json
import os

dout_split = '/root/data_alfred/splits'

split_entries = [
    {'task': 'look_at_obj_in_light-BaseballBat-None-DeskLamp-301/trial_T20200814_164125_595727', 
    'repeat_idx':0, 'full_traj_success': False, 'collected_subgoals': 2},
    {'task': 'pick_heat_then_place_in_recep-Potato-None-GarbageCan-27/trial_T20200817_012852_414204', 
    'repeat_idx':0, 'full_traj_success': False, 'collected_subgoals': 2},
    {'task': 'look_at_obj_in_light-Book-None-DeskLamp-319/trial_T20200814_165838_483790', 
    'repeat_idx':0, 'full_traj_success': False, 'collected_subgoals': 1},
    {'task': 'pick_and_place_simple-HandTowel-None-Toilet-414/trial_T20200815_151124_533360', 
    'repeat_idx':0, 'full_traj_success': False, 'collected_subgoals': 3},
    {'task': 'pick_and_place_with_movable_recep-Apple-Plate-CounterTop-3/trial_T20200814_214528_602344', 
    'repeat_idx':0, 'full_traj_success': False, 'collected_subgoals': 3},
    {'task': 'pick_and_place_with_movable_recep-Apple-Plate-CounterTop-30/trial_T20200814_214719_601048', 
    'repeat_idx':0, 'full_traj_success': False, 'collected_subgoals': 3},
    {'task': 'pick_and_place_with_movable_recep-Apple-Plate-DiningTable-11/trial_T20200814_220339_693684', 
    'repeat_idx':0, 'full_traj_success': False, 'collected_subgoals': 2},
    {'task': 'pick_clean_then_place_in_recep-Plate-None-Microwave-14/trial_T20200822_192653_991307', 
    'repeat_idx':0, 'full_traj_success': False, 'collected_subgoals': 3},    
    {'task': 'pick_clean_then_place_in_recep-Plate-None-Microwave-15/trial_T20200822_192847_378077', 
    'repeat_idx':0, 'full_traj_success': False, 'collected_subgoals': 3},
    {'task': 'pick_heat_then_place_in_recep-PotatoSliced-None-SinkBasin-26/trial_T20200817_024509_832045', 
    'repeat_idx':0, 'full_traj_success': False, 'collected_subgoals': 1},
    {'task': 'pick_clean_then_place_in_recep-Plate-None-Fridge-21/trial_T20200822_185547_091775', 
    'repeat_idx':0, 'full_traj_success': False, 'collected_subgoals': 1},
    {'task': 'pick_cool_then_place_in_recep-Pot-None-DiningTable-27/trial_T20200818_110753_373921', 
    'repeat_idx':0, 'full_traj_success': False, 'collected_subgoals': 5},
    {'task': 'pick_cool_then_place_in_recep-Pot-None-SinkBasin-6/trial_T20200818_122735_617736', 
    'repeat_idx':0, 'full_traj_success': False, 'collected_subgoals': 3},
    {'task': 'pick_two_obj_and_place-Cloth-None-SinkBasin-422/trial_T20200818_005722_610055', 
    'repeat_idx':0, 'full_traj_success': False, 'collected_subgoals': 6},    
    {'task': 'pick_two_obj_and_place-Cloth-None-Toilet-410/trial_T20200818_012616_074675', 
    'repeat_idx':0, 'full_traj_success': False, 'collected_subgoals': 3},
    {'task': 'pick_two_obj_and_place-CreditCard-None-ArmChair-211/trial_T20200818_021213_027494', 
    'repeat_idx':0, 'full_traj_success': True, 'collected_subgoals': 9}
]

split_path = os.path.join(dout_split, 'sample_failed_dummy_raw.json')
with open(split_path, 'w') as f:
    json.dump({'augmentation':split_entries}, f)




# with open(split_path, 'r') as f:
#     raw_split = json.load(f)


# pick_two_obj_and_place-CreditCard-None-ArmChair-211 \