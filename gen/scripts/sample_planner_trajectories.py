import os
import sys
sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'gen'))

import time
import multiprocessing as mp
import json
import random
import shutil
import argparse
import numpy as np
import pandas as pd
from collections import OrderedDict
from datetime import datetime

import constants
from env.thor_env import ThorEnv
from agents.deterministic_planner_agent import DeterministicPlannerAgent
from game_states.task_game_state_full_knowledge import TaskGameStateFullKnowledge
from utils.video_util import VideoSaver
from utils.dataset_management_util import load_successes_from_disk, load_fails_from_disk

# params
RAW_IMAGES_FOLDER = 'raw_images/'
DATA_JSON_FILENAME = 'traj_data.json'

# video saver
video_saver = VideoSaver()

# structures to help with constraint enforcement.
goal_to_required_variables = {"pick_and_place_simple": {"pickup", "receptacle", "scene"},
                              "pick_two_obj_and_place": {"pickup", "receptacle", "scene"},
                              "look_at_obj_in_light": {"pickup", "receptacle", "scene"},
                              "pick_clean_then_place_in_recep": {"pickup", "receptacle", "scene"},
                              "pick_heat_then_place_in_recep": {"pickup", "receptacle", "scene"},
                              "pick_cool_then_place_in_recep": {"pickup", "receptacle", "scene"},
                              "pick_and_place_with_movable_recep": {"pickup", "movable", "receptacle", "scene"}}
goal_to_pickup_type = {'pick_heat_then_place_in_recep': 'Heatable',
                       'pick_cool_then_place_in_recep': 'Coolable',
                       'pick_clean_then_place_in_recep': 'Cleanable'}
goal_to_receptacle_type = {'look_at_obj_in_light': "Toggleable"}
goal_to_invalid_receptacle = {'pick_heat_then_place_in_recep': {'Microwave'},
                              'pick_cool_then_place_in_recep': {'Fridge'},
                              'pick_clean_then_place_in_recep': {'SinkBasin'},
                              'pick_two_obj_and_place': {'CoffeeMachine', 'ToiletPaperHanger', 'HandTowelHolder'}}

scene_id_to_objs = {}
obj_to_scene_ids = {}
scenes_for_goal = {g: [] for g in constants.GOALS}
scene_to_type = {}


def make_task_name(task_tuple):
    gtype, pickup_obj, movable_obj, receptacle_obj, scene_num = task_tuple
    # 'pick_two_obj_and_place-Watch-None-Dresser-301'
    return '%s-%s-%s-%s-%d' % (gtype, pickup_obj, movable_obj, receptacle_obj, scene_num)

def create_dirs(task_name, seed, obj_repeat):
    '''
    create dir like 
    <args.save_path>/pick_two_obj_and_place-Watch-None-Dresser-301/trial_T20200609_122157_214995
    '''
    task_id = 'trial_T' + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    save_name = task_name + '/' + f'seed_{seed}_repeat_{obj_repeat}/' + task_id

    constants.save_path = os.path.join(constants.DATA_SAVE_PATH, save_name, RAW_IMAGES_FOLDER)
    if not os.path.exists(constants.save_path):
        os.makedirs(constants.save_path)

    print("Saving images to: " + constants.save_path)

def save_bookkeeping(task_name, save_path, traj_dirs, error_counts, traj_error_map):
    '''
    save successful and failed path strings to file.
    save error type and counts to file.
    '''
    timenow = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    # flatten to a list of success paths
    success_traj_dirs = []
    for seed_key in traj_dirs['successes']:
        success_traj_dirs += traj_dirs['successes'][seed_key]

    # save flat list of successful paths
    path_successes = os.path.join(save_path, f'{task_name}_success_dirs_T{timenow}.json') 
    with open(path_successes, 'w') as f:
        json.dump(success_traj_dirs, f)

    # save dictionary with both success and fails, along with their seeds.
    path_samp_res = os.path.join(save_path, f'{task_name}_sampled_traj_dirs_T{timenow}.json')
    with open(path_samp_res, 'w') as f:
        json.dump(traj_dirs, f)

    # save dictionary with errors and their counts
    path_errors = os.path.join(save_path, f'{task_name}_error_counts_T{timenow}.json')
    with open(path_errors, 'w') as f:
        json.dump(error_counts, f)

    # save dictionary with traj dir and their error messages
    path_errors = os.path.join(save_path, f'{task_name}_failed_dir_errors_T{timenow}.json')
    with open(path_errors, 'w') as f:
        json.dump(error_counts, f)

def sample_task_trajs(
    args, task_tuple, agent, env, obj_to_scene_ids, add_requirements=None):
    '''
    Sample trajectory according to task tuple, save to disk location.

    task_spec:  tuple(str(goal_type), str(pickup object type), str(movable receptacle object type),str(receptacle object type), int(scene number)). Example: ('pick_two_obj_and_place', 'Watch', 'None', 'Dresser', 205)

    add_requirements: optional dict. Example: {'obj_repeats_var':3, 'seed_var':42}
    '''
    # TODO multi-thread version  

    gtype, pickup_obj, movable_obj, receptacle_obj, scene_num = task_tuple
    print(f'Task: {task_tuple}')

    # success and failure book-keeping
    # k=error type, v=int count
    error_counts = {}
    # k=traj dir path, v=error message
    traj_error_map = {}
    # k=seed, v=traj dir path
    sampled_traj_dirs = {'successes':{}, 'fails':{}}

    # try multiple times
    tries_remaining = args.trials_before_fail
    num_place_fails = 0

    # set random seeds -- determines room object locations and agent start pos
    if 'seeds' in add_requirements.keys():
        seeds = add_requirements['seeds']
    else:
        seeds = [random.randint(0, 2 ** 32) for _ in range(args.trials_before_fail)]

    # optionally specify how 'crowded' the room is with non-task objects
    if 'obj_repeat' in add_requirements.keys():
        obj_repeat = add_requirements['obj_repeat']
    else:
        obj_repeat = None

    while tries_remaining > 0:

        constants.pddl_goal_type = gtype
        print("PDDLGoalType: " + constants.pddl_goal_type)            

        # determines room object locations and agent start pos
        seed = seeds.pop()
        # e.g. 'pick_two_obj_and_place-Watch-None-Dresser-205'
        task_name = make_task_name(task_tuple)
        # create task directory to store plan, trajectory json and raw images
        # e.g. <args.save_path>/pick_two_obj_and_place-Watch-None-Dresser-301/trial_T20200609_122157_214995
        task_id = create_dirs(task_name, seed, obj_repeat)

        # setup data dictionary for traj.json output
        setup_data_dict()
        constants.data_dict['task_id'] = task_id
        constants.data_dict['task_type'] = constants.pddl_goal_type
        constants.data_dict['dataset_params']['video_frame_rate'] = constants.VIDEO_FRAME_RATE

        try:

            # spawn pickup object instances
            # 'repeat', number of instance to spawn for pickup object type
            # 'sparse', how much space to free up around receptacle object instance
            constraint_objs = {'repeat': [(constants.OBJ_PARENTS[pickup_obj],
                                            np.random.randint(2 if gtype == "pick_two_obj_and_place" else 1,
                                                                constants.PICKUP_REPEAT_MAX + 1))],
                                'sparse': [(receptacle_obj.replace('Basin', ''),
                                            num_place_fails * constants.RECEPTACLE_SPARSE_POINTS)]}  

            # if task requires, spawn movable receptacle instances
            # 'repeat', number of instance to spawn for movable receptacle type, 
            if movable_obj != "None":
                constraint_objs['repeat'].append((movable_obj,
                                                    np.random.randint(1, constants.PICKUP_REPEAT_MAX + 1)))

            # spawn some more random objects in the scene
            # allow only object types listed in scene asset
            for obj_type in scene_id_to_objs[str(sampled_scene)]:
                # allow only object types not same as task objects
                if (obj_type in pickup_candidates and
                        obj_type != constants.OBJ_PARENTS[pickup_obj] and obj_type != movable_obj):
                    if obj_repeat is None:
                        constraint_objs['repeat'].append(
                            (obj_type,np.random.randint(1, constants.MAX_NUM_OF_OBJ_INSTANCES + 1)))
                    else:
                        constraint_objs['repeat'].append(
                            (obj_type, obj_repeat))

            # make sure there's enough space in microwave, sink, fridge etc if task needs it
            if gtype in goal_to_invalid_receptacle:
                constraint_objs['empty'] = [(r.replace('Basin', ''), num_place_fails * constants.RECEPTACLE_EMPTY_POINTS) for r in goal_to_invalid_receptacle[gtype]]

            # turn off the lights if task needs it
            constraint_objs['seton'] = []
            if gtype == 'look_at_obj_in_light':
                constraint_objs['seton'].append((receptacle_obj, False))

            # alert user that scene is now sparser if last try failed
            if num_place_fails > 0:
                print("Failed %d placements in the past; increased free point constraints: " % num_place_fails
                        + str(constraint_objs))           

            # thor env spawn up the scene according to constraint objs
            scene_info = {'scene_num': sampled_scene, 'random_seed': seed_var}
            info = agent.reset(scene=scene_info, objs=constraint_objs)

            # initialize problem definition for pddl planner
            task_objs = {'pickup': pickup_obj}
            if movable_obj != "None":
                task_objs['mrecep'] = movable_obj
            if gtype == "look_at_obj_in_light":
                task_objs['toggle'] = receptacle_obj
            else:
                task_objs['receptacle'] = receptacle_obj
            # specific object instances (with ID) are chosen for pickup and receptacle targets        
            agent.setup_problem({'info': info}, scene=scene_info, objs=task_objs)

            # start recording metadata for positions of objects
            object_poses = [{'objectName': obj['name'].split('(Clone)')[0],
                                'position': obj['position'],
                                'rotation': obj['rotation']}
                            for obj in env.last_event.metadata['objects'] if obj['pickupable']]
            dirty_and_empty = gtype == 'pick_clean_then_place_in_recep'
            object_toggles = [{'objectType': o, 'isOn': v}
                                for o, v in constraint_objs['seton']]
            constants.data_dict['scene']['object_poses'] = object_poses
            constants.data_dict['scene']['dirty_and_empty'] = dirty_and_empty
            constants.data_dict['scene']['object_toggles'] = object_toggles            

            # reinitialize the scene, in case THOR was updated, a different random seed wouldn't mess up these scenes.
            print("Performing reset via thor_env API")
            env.reset(sampled_scene)
            print("Performing restore via thor_env API")
            env.restore_scene(object_poses, object_toggles, dirty_and_empty)
            
            # send agent into the scene at an initial pos
            event = env.step(dict(constants.data_dict['scene']['init_action']))

            # compute the plan and execute it
            terminal = False
            while not terminal and agent.current_frame_count <= constants.MAX_EPISODE_LENGTH:
                # 1. agent get plan from solver -- ff_planner_handler.py get_plan_from_file()
                # 2. agent step in thor env -- plan_agent.py self.controller_agent.step()
                action_dict = agent.get_action(None)
                reward, terminal = agent.get_reward()

            # dump constants.data_dict to file
            dump_data_dict()
            # save images in images_path to video_path
            save_video()

            # stops trying once we succeed
            tries_remaining = 0
            
            # book keeping
            sampled_traj_dirs['successes'][seed] = constants.save_path

        except Exception as e:

            # report error in stdout
            import traceback
            traceback.print_exc()
            err_str = repr(e)
            print("Error: " + err_str)
            print("Invalid Task: skipping...")
            if args.debug:
                print(traceback.format_exc())
            
            # book keep errors to out files
            if err_str in error_counts.keys():
                error_counts[err_str] += 1
            else: 
                error_counts[err_str] = 1
            traj_error_map[constants.save_path] = err_str
            sampled_traj_dirs['fails'][seed] = constants.save_path

            num_place_fails += 1
            tries_remaining -= 1
            
            # if failed, save to save_path_fail to be examined or debugged

            pass
    
    return sampled_traj_dirs, error_counts, traj_error_map


def sample_task_params(args):
    # TODO
    # parse args
    # random sample 'random' inputs
    # avoid failure seeds or combinations
    # TODO specify additional requirements (like favorable seeds and object repeats)
    return (gtype, pickup_obj, movable_obj, receptacle_obj, scene_num), add_requirements


def main(args):

    # objects-to-scene and scene-to-objects database
    for scene_type, ids in constants.SCENE_TYPE.items():
        for id in ids:
            obj_json_file = os.path.join('layouts', 'FloorPlan%d-objects.json' % id)
            with open(obj_json_file, 'r') as of:
                scene_objs = json.load(of)

            id_str = str(id)
            scene_id_to_objs[id_str] = scene_objs
            for obj in scene_objs:
                if obj not in obj_to_scene_ids:
                    obj_to_scene_ids[obj] = set()
                obj_to_scene_ids[obj].add(id_str)

    # create env and agent
    env = ThorEnv()
    game_state = TaskGameStateFullKnowledge(env)
    agent = DeterministicPlannerAgent(thread_id=0, game_state=game_state)

    # construct valid task tuple
    task_tuple, add_requirements = sample_task_params(args)

    task_name = make_task_name(task_tuple)

    # call sample_task_trajs
    sampled_traj_dirs, error_counts, traj_error_map = sample_task_trajs(
        args, task_tuple, agent, env, obj_to_scene_ids, add_requirements)

    # save the directory paths for success and failed trajectories, 
    # and error counts to disk
    save_bookkeeping(
        task_name, args.save_path, 
        sampled_traj_dirs, error_counts, traj_error_map)


def parallel_main(args):
    procs = [mp.Process(target=main, args=(args,)) for _ in range(args.num_threads)]
    try:
        for proc in procs:
            proc.start()
            time.sleep(0.1)
    finally:
        for proc in procs:
            proc.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # save settings
    parser.add_argument('--force_unsave', action='store_true', help="don't save any data (for debugging purposes)")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--save_path', type=str, help="where to save the success & failure trajectories and data")

    # debugging settings
    parser.add_argument('--debug', action='store_true')
    parser.add_argument("--just_examine", action='store_true', help="just examine what data is gathered; don't gather more")

    # parser.add_argument('--x_display', type=str, required=False, default=constants.X_DISPLAY, help="x_display id")

    # multi-thread settings
    parser.add_argument("--in_parallel", action='store_true', help="this collection will run in parallel with others, so load from disk on every new sample")
    parser.add_argument("-n", "--num_threads", type=int, default=0, help="number of processes for parallel mode")
    parser.add_argument('--json_file', type=str, default="", help="path to json file with trajectory dump")

    # task params
    parser.add_argument("--goal", type=str, default='random', help='goal such as pick_two_obj_and_place. "random" for random pick.')
    parser.add_argument("--pickup", type=str, default='random', help='object name. "random" for random pick.')
    parser.add_argument("--movable", type=str, default='random', help='movable receptacle name. "random" for random pick.')
    parser.add_argument("--receptacle", type=str, default='random', help='receptacle name. "random" for random pick.')
    parser.add_argument("--scene", type=str, default='random', help='scene number. "999" for random pick.')
    parser.add_argument("--seed", type=int, default=999, help='scene number. "999" for random pick.')

    parser.add_argument("--trials_before_fail", type=int, default=5)

    parse_args = parser.parse_args()

    if parse_args.in_parallel and parse_args.num_threads > 1:
        parallel_main(parse_args)
    else:
        main(parse_args)
