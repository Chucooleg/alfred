'''From each sampled trajectories, we need to collect the environmental object states.'''

import os
import sys
sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'gen'))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'models'))

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

import time
import multiprocessing as mp
import subprocess

def load_task_json(args, task):
    '''
    load preprocessed json from disk
    ''' 
    json_path = os.path.join(args.data, task['task'], 'traj_data.json')

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
    def evaluate(cls, args, r_idx, env, split_name, traj_data, planner_full_traj_success, success_log_entries, fail_log_entries, results, logger):

        # setup scene
        reward_type = 'dense'
        cls.setup_scene(env, traj_data, r_idx, args, reward_type=reward_type)

        # --------------- collect actions -----------------
        # ground-truth low-level actions
        # e.g. ['LookDown_15', 'MoveAhead_25', 'MoveAhead_25', ... '<<stop>>']
        groundtruth_action_low = [a['discrete_action']['action'] for a in traj_data['plan']['low_actions']]
        groundtruth_action_low.append(cls.STOP_TOKEN)

        # get low-level action to high subgoal alignment
        # get valid interaction per low-level action
        # get interaction mask if any
        end_action = {
            'api_action': {'action': 'NoOp'},
            'discrete_action': {'action': '<<stop>>', 'args': {}},
            'high_idx': traj_data['plan']['high_pddl'][-1]['high_idx']
        }
        # e.g. [0,0,0, ... , 11], lenght = total T
        groundtruth_subgoal_alignment = []
        # e.g. [0,1,0, ... , 1], lenght = total T
        groundtruth_valid_interacts = []
        # len=num timestep with valid interact, np shape (1 , 300 , 300)
        groundtruth_low_mask = []
        for a in (traj_data['plan']['low_actions'] + [end_action]):
            # high-level action index (subgoals)
            groundtruth_subgoal_alignment.append(a['high_idx'])
            # interaction validity
            step_valid_interact = 1 if cls.has_interaction(a['discrete_action']['action']) else 0
            groundtruth_valid_interacts.append(step_valid_interact)
            # interaction mask values
            if 'mask' in a['discrete_action']['args'].keys() and a['discrete_action']['args']['mask'] is not None:
                groundtruth_low_mask.append(decompress_mask(a['discrete_action']['args']['mask']))

        # ground-truth high-level subgoals
        # e.g. ['GotoLocation', 'PickupObject', 'SliceObject', 'GotoLocation', 'PutObject', ... 'NoOp']
        groundtruth_action_high = [a['discrete_action']['action'] for a in traj_data['plan']['high_pddl']]
        
        assert len(groundtruth_action_low) == len(groundtruth_subgoal_alignment) == len(groundtruth_valid_interacts)
        assert len(groundtruth_action_high) == groundtruth_subgoal_alignment[-1] + 1
        assert sum(groundtruth_valid_interacts) == len(groundtruth_low_mask)

        # --------------- execute actions -----------------
        # initialize state dictionary for all timesteps
        states = []

        # get symbols and initial object states
        event = env.last_event
        obj_symbol_set = set(cls.get_object_symbols_present_in_scene(traj_data))
        receptacle_symbol_set = set(cls.get_receptacle_symbols_present_in_scene(event.metadata))
        object_states_last = cls.get_object_states(event.metadata) # includes receptacles

        # loop through actions and execute them in the sim env
        done, success = False, False
        fails = 0
        t = 0
        reward = 0
        action, mask = None, None
        interact_ct = 0
        high_idx = -1
        while not done:            
            # if last action was stop, break
            if action == cls.STOP_TOKEN:
                done = True
                logging.info("predicted STOP")
                break
            
            if high_idx < groundtruth_subgoal_alignment[t]:
                high_idx = groundtruth_subgoal_alignment[t]
                new_subgoal = True
            else:
                new_subgoal = False
            
            # collect metadata states only
            states.append({
                'new_subgoal': new_subgoal,
                'time_step': t,
                'subgoal_step': groundtruth_subgoal_alignment[t],
                'subgoal': groundtruth_action_high[groundtruth_subgoal_alignment[t]],
                'objects_metadata': event.metadata['objects'],
            })
            
            # collect groundtruth action and mask
            # single string
            action = groundtruth_action_low[t]
            # expect (300, 300)
            if groundtruth_valid_interacts[t]:
                mask = groundtruth_low_mask[interact_ct][0]
                interact_ct += 1
            else:
                mask = None

            # interact with the env
            t_success, event, _, err, _ = env.va_interact(action, interact_mask=mask, smooth_nav=args.smooth_nav)
            if not t_success:
                fails += 1
                if fails >= args.max_fails:
                    logging.info("Interact API failed %d times" % fails + "; latest error '%s'" % err)
                    break            
 
            # next time-step
            t_reward, t_done = env.get_transition_reward()
            reward += t_reward
            t += 1
        
        # make sure we have used all masks
        assert interact_ct == sum(groundtruth_valid_interacts)
        
        # check if goal was satisfied
        goal_satisfied = env.get_goal_satisfied()
        if goal_satisfied:
            print("Goal Reached")
            success = True

        # -------------------------------------------------
        # if the planner did not achieve full success, 
        # we need to remove the last collected state because the last action was not cls.STOP_TOKEN
        if not planner_full_traj_success:
            states.pop()

        # ------save the object states out --------------
        logging.info("Goal Reached")
        outpath = os.path.join(traj_data['raw_root'], 'metadata_states.json')
        logging.info('saving to outpath: {}'.format(outpath))
        with open(outpath, 'w') as f:
            json.dump(states, f)
        logging.info("----------------------------------------")

        return states, outpath

def main(args, splits_to_process_dict, process_i=0):

    raw_splits = splits_to_process_dict[process_i]

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_file_path = os.path.join(args.data, f'collect_demo_obj_states_T{args.PLANNER_TIME_STAMP}.log')
    hdlr = logging.FileHandler(log_file_path)
    logger.addHandler(hdlr)
    print (f'Logger is writing to {log_file_path}')

    # start sim env
    env = ThorEnv()
    
    # no language annotation available
    r_idx = None

    # book keeping -- some planner generated traj can still fail on execution
    # save to files
    failed_splits = {split_name:[] for split_name in raw_splits.keys()}
    out_splits = {split_name:[] for split_name in raw_splits.keys()}
    # report successes thus far (used in debugging only)
    success_log_entries = {split_name:[] for split_name in raw_splits.keys()}
    fail_log_entries = {split_name:[] for split_name in raw_splits.keys()}
    tot_ct = {split_name:len(raw_splits[split_name]) for split_name in raw_splits.keys()}
    results = {split_name:{} for split_name in raw_splits.keys()}

    print ('-----------START COLLECTING OBJECT STATES FROM RAW TRAJECTORIES-----------')
    for split_name in raw_splits.keys():
        tasks = [task for task in raw_splits[split_name]]
        split_count = 0
        print(f'Split {split_name} starts object states collection')
        print(f'Tasks: {tasks}')
        for task in progressbar.progressbar(tasks):
            traj_data = load_task_json(args, task)
            traj_data['raw_root'] = os.path.join(args.data, task['task'])
            planner_full_traj_success = task['full_traj_success']
            split_count += 1
            logger.info('-----------------')
            logger.info(f'Split {split_name}: {split_count}/{tot_ct[split_name]} task')
            logger.info(f'Task Root: {traj_data["raw_root"]}.')
            logger.info(f'Task Type: {traj_data["task_type"]}.')
            print(f'\nProcessing {traj_data["raw_root"]}')
            try:
                _, _ = CollectStates.evaluate(
                    args, r_idx, env, split_name, traj_data, planner_full_traj_success, 
                    success_log_entries, fail_log_entries, results, logger
                )
                print(f'Task succeeds to collect object state.')
                out_splits[split_name].append({
                    'task': task["task"], 
                    'repeat_idx':task['repeat_idx'], 
                    'full_traj_success':task['full_traj_success'],
                    'collected_subgoals':task['collected_subgoals']}) # '<goal type>/<task_id>'
                if args.first_task_only:
                    print(f"Found a successful traj for split {split_name}. Stopping for this split.")
                    break
            except Exception as e:
                print(e)
                failed_splits[split_name].append({'task': task["task"]})
                print(f'Task fails to collect object state.')
        print(
            f'Split {split_name} object states collection results: \
                successes={len(out_splits[split_name])}, fails={len(failed_splits[split_name])}, total={tot_ct[split_name]}'
        )
                                       
    # save success splits
    split_file_dir = '/'.join(args.raw_splits.split('/')[:-1])
    split_file_name = args.raw_splits.split('/')[-1] 
    out_splits_path = os.path.join(split_file_dir, split_file_name.replace('_raw.json', '.json'))
    with open(out_splits_path, 'w') as f:
        json.dump(out_splits, f)
    print(f'New split file for successful trajectories is saved to {out_splits_path}')

def parallel_main(args):
    procs = [
        mp.Process(target=main, args=(args, splits_to_process_dict, process_i)) \
            for process_i in range(args.num_processes)
    ]
    try:
        for proc in procs:
            proc.start()
            time.sleep(0.1)
    finally:
        for proc in procs:
            proc.join()
        subprocess.call(["pkill", "-f", 'thor'])


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        '--data', help='dataset directory.', type=str
    )
    parser.add_argument(
        '--raw_splits', help='json file containing raw trajectory splits.', type=str
    )
    parser.add_argument(
        '--reward_config', default='models/config/rewards.json', type=str
    )
    parser.add_argument(
        '--first_task_only', action='store_true', 
        help='only process the first task loaded for each split.'
    )

    # rollout
    parser.add_argument(
        '--max_fails', type=int, default=10, 
        help='max API execution failures before episode termination'
    )
    parser.add_argument(
        '--smooth_nav', dest='smooth_nav', action='store_true', 
        help='smooth nav actions (might be required based on training data)')

    # multi-process settings
    parser.add_argument(
        "--in_parallel", action='store_true', 
        help="this collection will run in parallel with others, so load from disk on every new sample"
    )
    parser.add_argument(
        "-n", "--num_processes", type=int, default=0, 
        help="number of processes for parallel mode"
    )

    parse_args = parser.parse_args()

    parse_args.reward_config = os.path.join(os.environ['ALFRED_ROOT'], parse_args.reward_config)
    parse_args.PLANNER_TIME_STAMP = datetime.now().strftime("%Y%m%d")

    # load splits
    with open(parse_args.raw_splits) as f:
        raw_splits = json.load(f)
    print(f'Raw Splits are : {raw_splits.keys()}')

    # divide task among processes
    # TODO: should replace with a proper queue for multi-process
    splits_to_process_dict = {}
    if parse_args.in_parallel and parse_args.num_processes > 1:
        quotient = len(raw_splits['augmentation']) // parse_args.num_processes
        for process_i in range(parse_args.num_processes):
            splits_to_process_dict[process_i] = {
                'augmentation': raw_splits['augmentation'][process_i*quotient: (process_i+1)*quotient]
            }
            if process_i == parse_args.num_processes-1:
                splits_to_process_dict[process_i]['augmentation'] += raw_splits['augmentation'][(process_i+1)*quotient:]

        parallel_main(parse_args)
    else:
        splits_to_process_dict[0] = raw_splits
        main(parse_args, splits_to_process_dict)

