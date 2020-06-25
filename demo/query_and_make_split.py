import os
import json
import pickle
import random
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import time
from datetime import datetime

save = pickle.load( open( "demo/task_lookup.p", "rb" ) )

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--data', help='dataset folder', default='/root/data_alfred/json_feat_2.1.0/')
    parser.add_argument('--splits', help='dataset folder', default='/root/data_alfred/splits/')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    print('\n\n\nWelcome to the skill-learning demo.\n')

    # 1. choose skill
    skill_int = 999
    while not skill_int in range(0,8):
        print('\nPlease choose a skill:')
        for i in range(0,8):
            print (f'{i}. {save["skill_set"][i]}')
        skill_int = input('\nSkill choice (Valid Number Only): ')
        skill_int = int(skill_int)
    if skill_int == 0:
        skill_int = random.randint(1, 7) 
    print (f'Skill = {save["skill_set"][skill_int]}')

    # 2. choose pickup object
    idxs = sorted([int(idx) for idx in save['lookup'][str(skill_int)].keys()])
    pickupObject_int = 999
    while not pickupObject_int in [0] + idxs:
        print('\nPlease choose a pickup object:')
        for idx in [0] + idxs:
            print(f'{idx}. {save["pickupObject_set"][idx]}')
        pickupObject_int = input('\nPickup Object choice (Valid Number Only): ')
        pickupObject_int = int(pickupObject_int)
    if pickupObject_int == 0:
        pickupObject_int = random.choice(idxs)
    print (f'Pickup Object = {save["pickupObject_set"][pickupObject_int]}')

    # 3. choose movable receptacle
    if save["skill_set"][skill_int] == 'pick_and_place_with_movable_recep':
        idxs = sorted([int(idx) for idx in save['lookup'][str(skill_int)][str(pickupObject_int)].keys()])
        movable_int = 999
        while not movable_int in [0] + idxs:
            print('\nPlease choose a movable receptacle:')
            for idx in [0] + idxs:
                print(f'{idx}. {save["movable_set"][idx]}')
            movable_int = input('\nMovable Receptacle choice (Valid Number Only): ')
            movable_int = int(movable_int)
        if movable_int == 0:
            movable_int = random.choice(idxs)
        print (f'Movable Receptacle = {save["movable_set"][movable_int]}')
    else:
        movable_int = save["movable2num"]['None']
        print (f'Movable Receptacle = {save["movable_set"][movable_int]}')

    # 4. choose a final receptacle
    idxs = sorted([int(idx) for idx in save['lookup'][str(skill_int)][str(pickupObject_int)][str(movable_int)].keys()])
    receptacle_int = 999
    while not receptacle_int in [0] + idxs:
        print('\nPlease choose a final receptacle:')
        for idx in [0] + idxs:
            print(f'{idx}. {save["receptacle_set"][idx]}')
        receptacle_int = input('\nFinal Receptacle choice (Valid Number Only): ')
        receptacle_int = int(receptacle_int)
    if receptacle_int == 0:
        receptacle_int = random.choice(idxs)
    print (f'Final Receptacle = {save["receptacle_set"][receptacle_int]}')

    # 5. choose a scene
    idxs = sorted([int(idx) for idx in save['lookup'][str(skill_int)][str(pickupObject_int)][str(movable_int)][str(receptacle_int)].keys()])
    print(f'\nAvailable scene numbers : {idxs}')
    scene_num = random.choice(idxs)
    print(f'Sampled random scene {scene_num}.')

    # print tuple
    task_tuple = (save["skill_set"][skill_int], save["pickupObject_set"][pickupObject_int], save["movable_set"][movable_int], save["receptacle_set"][receptacle_int], scene_num)
    if args.debug:
        print(f'\nTask tuple: {task_tuple}' )

    # make split
    # {'valid_seen': [T_..., T_...], 'valid_unseen': [T_..., T_...]}
    splits_and_trajs = save['lookup'][str(skill_int)][str(pickupObject_int)][str(movable_int)][str(receptacle_int)][str(scene_num)]
    split = random.choice(list(splits_and_trajs.keys()))
    traj_id = random.choice(splits_and_trajs[split])
    if args.debug:
        print( split, traj_id)

    # return a video link    
    task_name = '{}-{}-{}-{}-{}'.format(save["skill_set"][skill_int], save["pickupObject_set"][pickupObject_int], save["movable_set"][movable_int], save["receptacle_set"][receptacle_int], scene_num)
    url = f'https://mturk.jessethomason.com/lang_2_plan/2.1.0/{task_name}/{traj_id}/video.mp4'
    print (f'\n\n\nClick to watch Planner Sampled Trajectory:\n {url}\n\n\n\n\n\n\n')

    # make new split
    new_split = {split: [{'task': f'{task_name}/{traj_id}', 'repeat_idx':0}]}
    TIME_NOW = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    new_split_path = os.path.join(args.splits, f'demo_T{TIME_NOW}.json')
    with open(new_split_path, 'w') as f:
        json.dump(new_split, f)
    if args.debug:
        print('Made new split file',new_split_path)
    time.sleep(2)
    print('-------------------------------------------------------------------------------')
