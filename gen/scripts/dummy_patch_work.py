import json
import os

# -------------------------------------------------------------------------------# 
# MAKE TOY SPLIT

# dout_split = '/root/data_alfred/splits'

# split_entries = [
#     {'task': 'look_at_obj_in_light-BaseballBat-None-DeskLamp-301/trial_T20200814_164125_595727', 
#     'repeat_idx':0, 'full_traj_success': False, 'collected_subgoals': 2},
#     {'task': 'pick_heat_then_place_in_recep-Potato-None-GarbageCan-27/trial_T20200817_012852_414204', 
#     'repeat_idx':0, 'full_traj_success': False, 'collected_subgoals': 2},
#     {'task': 'look_at_obj_in_light-Book-None-DeskLamp-319/trial_T20200814_165838_483790', 
#     'repeat_idx':0, 'full_traj_success': False, 'collected_subgoals': 1},
#     {'task': 'pick_and_place_simple-HandTowel-None-Toilet-414/trial_T20200815_151124_533360', 
#     'repeat_idx':0, 'full_traj_success': False, 'collected_subgoals': 3},
#     {'task': 'pick_and_place_with_movable_recep-Apple-Plate-CounterTop-3/trial_T20200814_214528_602344', 
#     'repeat_idx':0, 'full_traj_success': False, 'collected_subgoals': 3},
#     {'task': 'pick_and_place_with_movable_recep-Apple-Plate-CounterTop-30/trial_T20200814_214719_601048', 
#     'repeat_idx':0, 'full_traj_success': False, 'collected_subgoals': 3},
#     {'task': 'pick_and_place_with_movable_recep-Apple-Plate-DiningTable-11/trial_T20200814_220339_693684', 
#     'repeat_idx':0, 'full_traj_success': False, 'collected_subgoals': 2},
#     {'task': 'pick_clean_then_place_in_recep-Plate-None-Microwave-14/trial_T20200822_192653_991307', 
#     'repeat_idx':0, 'full_traj_success': False, 'collected_subgoals': 3},    
#     {'task': 'pick_clean_then_place_in_recep-Plate-None-Microwave-15/trial_T20200822_192847_378077', 
#     'repeat_idx':0, 'full_traj_success': False, 'collected_subgoals': 3},
#     {'task': 'pick_heat_then_place_in_recep-PotatoSliced-None-SinkBasin-26/trial_T20200817_024509_832045', 
#     'repeat_idx':0, 'full_traj_success': False, 'collected_subgoals': 1},
#     {'task': 'pick_clean_then_place_in_recep-Plate-None-Fridge-21/trial_T20200822_185547_091775', 
#     'repeat_idx':0, 'full_traj_success': False, 'collected_subgoals': 1},
#     {'task': 'pick_cool_then_place_in_recep-Pot-None-DiningTable-27/trial_T20200818_110753_373921', 
#     'repeat_idx':0, 'full_traj_success': False, 'collected_subgoals': 5},
#     {'task': 'pick_cool_then_place_in_recep-Pot-None-SinkBasin-6/trial_T20200818_122735_617736', 
#     'repeat_idx':0, 'full_traj_success': False, 'collected_subgoals': 3},
#     {'task': 'pick_two_obj_and_place-Cloth-None-SinkBasin-422/trial_T20200818_005722_610055', 
#     'repeat_idx':0, 'full_traj_success': False, 'collected_subgoals': 6},    
#     {'task': 'pick_two_obj_and_place-Cloth-None-Toilet-410/trial_T20200818_012616_074675', 
#     'repeat_idx':0, 'full_traj_success': False, 'collected_subgoals': 3},
#     {'task': 'pick_two_obj_and_place-CreditCard-None-ArmChair-211/trial_T20200818_021213_027494', 
#     'repeat_idx':0, 'full_traj_success': True, 'collected_subgoals': 9}]


# split_path = os.path.join(dout_split, 'sample_failed_dummy_raw.json')
# with open(split_path, 'w') as f:
#     json.dump({'augmentation':split_entries}, f)



# -------------------------------------------------------------------------------# 
# CORRECT MULTI-THREADING MISTAKE

# # 8851
# raw_splits_p = '/root/data_alfred/splits/sample_failed_20200820_raw.json'
# # 7649
# splits_p = '/root/data_alfred/splits/sample_failed_20200820.json'

# data = '/root/data_alfred/json_data_augmentation_20200820'

# with open(raw_splits_p, 'r') as f:
#     raw_splits = json.load(f)

# print('RAW', len(raw_splits['augmentation']))

# splits = {'augmentation':[]}
# for task in raw_splits['augmentation']:
#     if os.path.exists(os.path.join(data, task['task'], 'metadata_states.json')):
#         splits['augmentation'].append(task)

# print('FINAL', len(splits['augmentation']))
# print (splits['augmentation'][0])

# with open(splits_p, 'w') as f:
#     json.dump(splits, f)


# pick_two_obj_and_place-CreditCard-None-ArmChair-211 \
# -------------------------------------------------------------------------------# 
# correct the ann key for baseline vs explainer outputs

import progressbar

# splits_p = '/root/data_alfred/splits/sample_failed_20200820_filtered.json'
# data = '/root/data_alfred/json_data_augmentation_20200820'

splits_p = '/root/data_alfred/splits/sample_failed_dummy.json'
data = '/root/data_alfred/json_dummy'

with open(splits_p, 'r') as f:
    splits = json.load(f)

baseline = 0
explainer = 0
none = 0

for task in progressbar.progressbar(splits['augmentation']):

    traj_data_p = os.path.join(data, task['task'], 'traj_data.json')
    with open(traj_data_p, 'r') as f:
        traj_data = json.load(f)
    
    # print(traj_data.keys())
    if 'baseline_annotations' in traj_data.keys():
        baseline += 1
    if 'explainer_annotations' in traj_data.keys():
        explainer += 1
    if not 'baseline_annotations' in traj_data.keys() and not 'explainer_annotations' in traj_data.keys():
        none += 1
    # traj_data['baseline_annotations'] = traj_data['explainer_annotations'].copy()
    # del traj_data['explainer_annotations']

    # with open(traj_data_p, 'w') as f:
    #     json.dump(traj_data, f)

print (baseline, explainer, none)
# -------------------------------------------------------------------------------# 