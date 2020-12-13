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

import time
import multiprocessing as mp
import subprocess


parser = ArgumentParser()

# data
parser.add_argument('--data', help='dataset folder', default='/root/data_alfred/demo_generated/new_trajectories')
parser.add_argument('--raw_splits', help='json file containing raw splits coming directly out from planner.', default='/root/data_alfred/splits/demo_june13_raw.json')
parser.add_argument('--reward_config', default='models/config/rewards.json')
parser.add_argument('--first_task_only', action='store_true', help='only process the first task loaded for each split.')

# rollout
parser.add_argument('--max_fails', type=int, default=10, help='max API execution failures before episode termination')
parser.add_argument('--subgoals', type=str, help="subgoals to evaluate independently, eg:all or GotoLocation,PickupObject...", default="")
parser.add_argument('--smooth_nav', dest='smooth_nav', action='store_true', help='smooth nav actions (might be required based on training data)')

# multi-thread settings
parser.add_argument("--in_parallel", action='store_true', help="this collection will run in parallel with others, so load from disk on every new sample")
parser.add_argument("-n", "--num_threads", type=int, default=0, help="number of processes for parallel mode")

# debug
parser.add_argument('--debug', dest='debug', action='store_true') # TODO True will give rise to X DISPLAY ERROR
parse_args = parser.parse_args()

parse_args.reward_config = os.path.join(os.environ['ALFRED_ROOT'], parse_args.reward_config)
# parse_args.PLANNER_TIME_STAMP = re.findall('new_trajectories_T(.*)/', parse_args.data)[0]
parse_args.PLANNER_TIME_STAMP = '20201210'


def load_task_json(args, task):
    '''
    load preprocessed json from disk
    ''' 
    # e.g. /root/data_alfred/demo_generated/new_trajectories_debug_sampler_20200611/pick_two_obj_and_place-Watch-None-Dresser-205/trial_T20200611_235502_613792/traj_data.json
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

def evaluate(env, traj_data, r_idx, parse_args):
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

        # -------------------------------------------------
        # --------------- execute actions -----------------

        # get symbols and initial object states
        event = env.last_event

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
            if action == "<<stop>>":
                done = True
                logging.info("predicted STOP")
                break
            
            if high_idx < groundtruth_subgoal_alignment[t]:
                high_idx = groundtruth_subgoal_alignment[t]
                new_subgoal = True
            else:
                new_subgoal = False
            
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
            t_success, event, _, err, _ = env.va_interact(action, interact_mask=mask, smooth_nav=args.smooth_nav, debug=args.debug)
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


# Logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
log_file_path = os.path.join(parse_args.data, f'collect_demo_obj_states_T{parse_args.PLANNER_TIME_STAMP}.log')
hdlr = logging.FileHandler(log_file_path)
logger.addHandler(hdlr)
print (f'Logger is writing to {log_file_path}')

# Failed Task
p = '/root/data_alfred/unlabeled_12k_20201206/seen/pick_clean_then_place_in_recep-PotatoSliced-None-GarbageCan-27/trial_T20190918_153044_946925/traj_data.json'
with open(p, 'r') as f:
    traj_data = json.load(f)

# Set env
env = ThorEnv()
r_idx = 0

# 

evaluate(parse_args, r_idx, env, 'augmentation', traj_data)

