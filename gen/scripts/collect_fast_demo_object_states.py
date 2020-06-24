import os
import sys
sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'gen'))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'models'))

import torch
import pprint
import json
from data.preprocess import Dataset
from importlib import import_module, reload
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from models.utils.helper_utils import optimizer_to
from gen.utils.image_util import decompress_mask as util_decompress_mask
import gen.constants as constants

import re
import numpy as np
from PIL import Image
from datetime import datetime
from models.eval.eval import Eval
from env.thor_env import ThorEnv
from models.eval.eval_task import EvalTask
from collections import defaultdict
import logging
import progressbar

import shutil

def load_task_json(args, split_name, task):
    '''
    load preprocessed json from disk
    ''' 
    # e.g. /root/data_alfred/demo_generated/new_trajectories_debug_sampler_20200611/pick_two_obj_and_place-Watch-None-Dresser-205/trial_T20200611_235502_613792/traj_data.json
    json_path = os.path.join(args.data, split_name, task['task'], 'traj_data.json')

    with open(json_path) as f:
        data = json.load(f)
    return data

def decompress_mask(compressed_mask):
    '''
    decompress mask from json files
    '''
    mask = np.array(util_decompress_mask(compressed_mask))
    mask = np.expand_dims(mask, axis=0)
    return mask

class CollectStates(EvalTask):

    object_state_list = ['isToggled', 'isBroken', 'isFilledWithLiquid', 'isDirty',
                  'isUsedUp', 'isCooked', 'ObjectTemperature', 'isSliced',
                  'isOpen', 'isPickedUp', 'mass', 'receptacleObjectIds']

    object_symbol_list = constants.OBJECTS

    @classmethod
    def get_object_list(cls, traj_data):
        object_list = [ob['objectName'] for ob in traj_data['scene']['object_poses']]
        for ob in object_list:
            assert ob.split('_')[0] in constants.OBJECTS
        return object_list

    @classmethod
    def get_object_states(cls, metadata):
        object_states = defaultdict(dict)
        for ob in metadata['objects']:
            symbol = ob['name'].split('_')[0]
            # assert symbol in cls.object_symbol_list
            object_states[ob['name']]['symbol'] = symbol
            object_states[ob['name']]['states'] = {state:ob[state] for state in cls.object_state_list}
            object_states[ob['name']]['states']['parentReceptacles'] = ob['parentReceptacles'][0].split('|')[0] if ob['parentReceptacles'] is not None else None
        return object_states

    @classmethod    
    def divide_objects_by_change(cls, object_states_curr, object_states_last):
        objects_unchanged = []
        objects_changed = []
        for ob_name in object_states_last.keys():
            changed = False
            for state in cls.object_state_list + ['parentReceptacles']:
                if state in object_states_last[ob_name]['states'].keys():
                    if object_states_last[ob_name]['states'][state] != object_states_curr[ob_name]['states'][state]:
                        changed = True
            if changed == False:
                objects_unchanged.append(ob_name)
            else:
                objects_changed.append(ob_name)
        return objects_changed, objects_unchanged

    @classmethod  
    def get_unchanged_symbols(cls, objects_changed, objects_unchanged, symbol_set):
        objects_symbols_changed = [ob_name.split('_')[0] for ob_name in objects_changed]
        objects_symbols_unchanged = [ob_name.split('_')[0] for ob_name in objects_unchanged]
        return list((set(objects_symbols_unchanged) - set(objects_symbols_changed)) & symbol_set)

    @classmethod
    def get_object_symbols_present_in_scene(cls, traj_data):
        object_list = [ob['objectName'] for ob in traj_data['scene']['object_poses']]
        extracted_symbols = [ob.split('_')[0] for ob in object_list]
        # for symbol in extracted_symbols:
        #     assert symbol in cls.object_symbol_list
        return extracted_symbols

    @classmethod
    def get_receptacle_symbols_present_in_scene(cls, metadata):
        receptacle_list = [ob['name'] for ob in metadata['objects'] if ob['receptacle']]
        extracted_symbols = [ob.split('_')[0] for ob in receptacle_list]
        return extracted_symbols

    @classmethod
    def get_visibility(cls, metadata, object_symbols, receptacle_symbols):
        visible_objects = {ob:False for ob in object_symbols}
        visible_receptacles = {recp:False for recp in receptacle_symbols}
        for ob in metadata['objects']:
            if ob['visible']:
                symbol = ob['name'].split('_')[0]
                if ob['receptacle']:
                    visible_receptacles[symbol] = True
                else:
                    visible_objects[symbol] = True
        return [ob for ob in visible_objects.keys() if visible_objects[ob]], [recp for recp in visible_receptacles.keys() if visible_receptacles[recp]]    

    @classmethod
    def has_interaction(cls, action):
        '''
        check if low-level action is interactive
        '''
        non_interact_actions = ['MoveAhead', 'Rotate', 'Look', '<<stop>>', '<<pad>>', '<<seg>>']
        if any(a in action for a in non_interact_actions):
            return False
        else:
            return True        

    @classmethod
    def copy_metadata_to_raw(cls, args, r_idx, split_name, traj_data, success_log_entries, fail_log_entries, results, logger):
        old_path = os.path.join(traj_data['old_root'], 'pp', 'metadata_states.json')
        new_path = os.path.join(traj_data['root'], 'metadata_states.json')
        shutil.copy(src=old_path, dst=new_path)

def main(args):

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_file_path = os.path.join(args.data, f'collect_all_obj_states_T{"20200623"}.log')
    hdlr = logging.FileHandler(log_file_path)
    logger.addHandler(hdlr)
    print (f'Logger is writing to {log_file_path}')
    
    # load splits
    with open(args.splits) as f:
        splits = json.load(f)
    splits = {k:splits[k] for k in splits.keys() if k in ['train', 'valid_seen', 'valid_unseen']}
    print(f'Splits are : {splits.keys()}')

    # no language annotation available
    r_idx = 0

    # book keeping -- some planner generated traj can still fail on execution
    # save to files
    failed_splits = {split_name:[] for split_name in splits.keys()}
    out_splits = {split_name:[] for split_name in splits.keys()}
    # report successes thus far (used in debugging only)
    success_log_entries = {split_name:[] for split_name in splits.keys()}
    fail_log_entries = {split_name:[] for split_name in splits.keys()}
    tot_ct = {split_name:len(splits[split_name]) for split_name in splits.keys()}
    results = {split_name:{} for split_name in splits.keys()}

    # loop through splits
    print ('-----------START COLLECTING OBJECT STATES FROM TRAJECTORIES-----------')
    for split_name in splits.keys():
        tasks = [task for task in splits[split_name]]
        split_count = 0
        print(f'Split {split_name} starts object states collection')
        for task in progressbar.progressbar(tasks):
            traj_data = load_task_json(args, split_name, task)
            traj_data['old_root'] = os.path.join(args.data, task['task'])
            traj_data['root'] = os.path.join(args.data, split_name, task['task'])
            split_count += 1
            logger.info('-----------------')
            logger.info(f'Split {split_name}: {split_count}/{tot_ct[split_name]} task')
            logger.info(f'Task Root: {traj_data["root"]}.')
            logger.info(f'Task Type: {traj_data["task_type"]}.')
            print(f'\nProcessing {traj_data["root"]}')
            try:
                CollectStates.copy_metadata_to_raw(args, r_idx, split_name, traj_data, success_log_entries, fail_log_entries, results, logger)
                print(f'Task succeeds to copy object state.')
                out_splits[split_name].append({'task': task["task"], 'repeat_idx':task['repeat_idx']}) # '<goal type>/<task_id>'
            except Exception as e:
                print(e)
                failed_splits[split_name].append({'task': (task["task"], str(e))})
                print(f'Task fails to copy object state.')
        print(f'Split {split_name} object states collection results: successes={len(out_splits[split_name])}, fails={len(failed_splits[split_name])}, total={tot_ct[split_name]}')
                                       
    # save success splits
    # /root/data_alfred/splits/
    # split_file_dir = '/'.join(args.splits.split('/')[:-1])
    # may17.json
    # split_file_name = args.splits.split('/')[-1] 
    # /root/data_alfred/splits/demo_june13.json
    # out_splits_path = os.path.join(split_file_dir, split_file_name.replace('may17.json', 'june23.json'))
    # with open(out_splits_path, 'w') as f:
    #     json.dump(out_splits, f)
    # print(f'New split file for successful trajectories is saved to {out_splits_path}')

    # # save failed splits if debuggin
    # if args.debug:
    #     # save failed splits
    #     # /root/data_alfred/splits/demo_june13_failed.json
    #     failed_splits_path = os.path.join(split_file_dir, split_file_name.replace('may17.json', 'june23_failed.json'))
    #     with open(failed_splits_path, 'w') as f:
    #         json.dump(failed_splits, f)
    #     print(f'New split file for failed trajectories is saved to {failed_splits_path}')

if __name__ == "__main__":
    parser = ArgumentParser()

    # data
    parser.add_argument('--data', help='dataset folder', default='/root/data_alfred/json_feat_2.1.0')
    parser.add_argument('--splits', help='json file containing raw splits coming directly out from planner.', default='/root/data_alfred/splits/may17.json')
    parser.add_argument('--reward_config', default='models/config/rewards.json')

    # rollout
    parser.add_argument('--max_fails', type=int, default=10, help='max API execution failures before episode termination')
    parser.add_argument('--subgoals', type=str, help="subgoals to evaluate independently, eg:all or GotoLocation,PickupObject...", default="")
    parser.add_argument('--smooth_nav', dest='smooth_nav', action='store_true', help='smooth nav actions (might be required based on training data)')

    # debug
    parser.add_argument('--debug', dest='debug', action='store_true') # TODO True will give rise to X DISPLAY ERROR
    parse_args = parser.parse_args()

    parse_args.reward_config = os.path.join(os.environ['ALFRED_ROOT'], parse_args.reward_config)
    # parse_args.PLANNER_TIME_STAMP = re.findall('new_trajectories_T(.*)/', parse_args.data)[0]

    main(parse_args)
    
# --------------------------------------------------------------------------------------------
# export ALFRED_ROOT=/root/data/home/hoyeung/alfred
# export DATA=/root/data_alfred/json_feat_2.1.0
# export SPLITS=/root/data_alfred/splits/may17.json
# cd $ALFRED_ROOT/gen
# python scripts/collect_fast_demo_object_states.py --data $DATA --splits $SPLITS