'''Sample partial/full trajectories starting from a list of previously failed trajectories.'''

import os
import sys
sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'gen'))

import time
import multiprocessing as mp
import subprocess
import json
import random
import shutil
import argparse
import numpy as np
import progressbar
import pandas as pd
from collections import OrderedDict
from datetime import datetime

import constants
from constants_scene_nums import VALID_UNSEEN_SCENES, TEST_SEEN_SCENES, TEST_UNSEEN_SCENES
from env.thor_env import ThorEnv
from agents.deterministic_planner_agent import DeterministicPlannerAgent
from game_states.task_game_state_full_knowledge import TaskGameStateFullKnowledge
from utils.video_util import VideoSaver


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


def save_video():
    images_path = constants.save_path + '*.png'
    video_path = os.path.join(constants.save_path.replace(RAW_IMAGES_FOLDER, ''), 'video.mp4')
    video_saver.save(images_path, video_path)

def setup_data_dict():
    constants.data_dict = OrderedDict()
    constants.data_dict['task_id'] = ""
    constants.data_dict['task_type'] = ""
    constants.data_dict['scene'] = {'floor_plan': "", 'random_seed': -1, 'scene_num': -1, 'init_action': [],
                                    'object_poses': [], 'dirty_and_empty': None, 'object_toggles': []}
    constants.data_dict['plan'] = {'high_pddl': [], 'low_actions': []}
    constants.data_dict['images'] = []
    constants.data_dict['template'] = {'task_desc': "", 'high_descs': []}
    constants.data_dict['pddl_params'] = {'object_target': -1, 'object_sliced': -1,
                                          'parent_target': -1, 'toggle_target': -1,
                                          'mrecep_target': -1}
    constants.data_dict['dataset_params'] = {'video_frame_rate': -1}
    constants.data_dict['pddl_state'] = []

def dump_data_dict():
    data_save_path = constants.save_path.replace(RAW_IMAGES_FOLDER, '')
    with open(os.path.join(data_save_path, DATA_JSON_FILENAME), 'w') as fp:
        json.dump(constants.data_dict, fp, sort_keys=True, indent=4)

def delete_save(in_parallel):
    '''
    Delete all collected images in save directory. Used in debug runs.
    in_parallel: boolean. If using multiple processes and another processes is attempting this task, 
                 don't delete.
    '''
    save_folder = constants.save_path.replace(RAW_IMAGES_FOLDER, '')
    if os.path.exists(save_folder):
        try:
            shutil.rmtree(save_folder)
        except OSError as e:
            if in_parallel:  # another process succeeded at this task while this one failed.
                return False
            else:
                raise e  # if we're not running in parallel, this is an actual.
    return True

def make_task_name(task_tuple):
    gtype, pickup_obj, movable_obj, receptacle_obj, scene_num = task_tuple
    # 'pick_two_obj_and_place-Watch-None-Dresser-301'
    return '%s-%s-%s-%s-%s' % (gtype, pickup_obj, movable_obj, receptacle_obj, scene_num)

def create_dirs(task_name, seed, obj_repeat):
    '''
    create dir like 
    <args.save_path>/pick_two_obj_and_place-Watch-None-Dresser-301/trial_T20200609_122157_214995
    '''
    task_id = 'trial_T' + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    # pick_cool_then_place_in_recep-LettuceSliced-None-DiningTable-17/traj_T_...../
    task_name_task_id = f'{task_name}/{task_id}/'

    # constants.DATA_SAVE_PATH -- /data_alfred/sampled/new_trajectories_T.../
    constants.save_path = os.path.join(constants.DATA_SAVE_PATH, task_name_task_id, RAW_IMAGES_FOLDER)
    if not os.path.exists(constants.save_path):
        os.makedirs(constants.save_path)

    print("Saving images to: " + constants.save_path)
    return task_id

def save_bookkeeping_and_splits(task_name, save_path, splits_dir, traj_dirs, errors, process_i=None):
    '''
    save successful and failed path strings to file.
    save error type and counts to file.
    '''

    thread_suffix = f'_{process_i}' if process_i is not None else ''

    # flatten to a list of success paths
    success_traj_dirs = []
    # make raw split file for nex step in pipeline -- object state collection
    split_entries = []
    for seed_key in traj_dirs['successes']:
        # append e.g. /data_alfred/sampled/new_trajectories_T.../pick_two_obj_and_place-Watch-None-Dresser-301/trial_T20200609_122157_214995
        success_traj_dirs.append(traj_dirs['successes'][seed_key])
        # append e.g. {'task': 'pick_two_obj_and_place-Watch-None-Dresser-301/trial_T20200609_122157_214995'}
        split_entries.append({'task':'/'.join(traj_dirs['successes'][seed_key].split('/')[-3:-1])})

    # save flat list of successful paths
    # /data_alfred/sampled/new_trajectories_T..../<task_name>_success_dirs_T.....json
    path_successes = os.path.join(
        save_path, f'{task_name}_success_dirs_T{constants.TIME_NOW}{thread_suffix}.json'
        ) 
    with open(path_successes, 'w') as f:
        json.dump(success_traj_dirs, f)

    # save dictionary with both success and fails, along with their seeds.
    path_samp_res = os.path.join(
        save_path, f'{task_name}_sampled_traj_dirs_T{constants.TIME_NOW}{thread_suffix}.json'
        )
    with open(path_samp_res, 'w') as f:
        json.dump(traj_dirs, f)

    # save dictionary with errors and their counts
    if process_i is not None:
        path_errors = os.path.join(
            save_path, f'{task_name}_errors_T{constants.TIME_NOW}{thread_suffix}.json'
            )
        with open(path_errors, 'w') as f:
            json.dump(errors, f)
    else:
        path_errors = os.path.join(save_path, f'errors_T{constants.TIME_NOW}.json')
        with open(path_errors, 'w') as f:
            json.dump({task_name: errors}, f)        

    # save to raw split file
    # /data_alfred/splits/demo_T....._raw.json
    split_path = os.path.join(splits_dir, f'demo_T{constants.TIME_NOW}{thread_suffix}_raw.json')
    with open(split_path, 'w') as f:
        json.dump({'demo':split_entries}, f)


def merge_process_results(args):
    '''merge results from multiple threads, they may or may not be working on the same task tuples'''

    # Merge splits
    merge_split = {'demo':[]}
    output_split_path = os.path.join(args.splits_dir, f'demo_T{constants.TIME_NOW}_raw.json')
    for process_i in range(args.num_processes):
        thread_split_path = output_split_path.replace('_raw', f'_{process_i}_raw')
        with open(thread_split_path, 'r') as f:
            merge_split['demo'] += json.load(f)['demo']
    with open(output_split_path, 'w') as f:
        json.dump(merge_split, f)
    print('\nSaved output split to :', output_split_path)
    
    # Merge error counts
    # identify all the task names
    task_names = []
    # e.g. /data_alfred/sampled/new_trajectories_T20200615_203135_456358
    for root, dirs, files in os.walk(constants.DATA_SAVE_PATH):
        for d in dirs:
            if d.count('-') == 4:
                task_names.append(d)
    task_names = set(task_names)
    print('Task names from threads:', task_names)

    merge_errors = {}
    output_errors_path = os.path.join(constants.DATA_SAVE_PATH, f'errors_T{constants.TIME_NOW}.json')
    for task_name in task_names:
        merge_errors[task_name] = {}
        for process_i in range(args.num_processes):
            thread_errors_path = os.path.join(
                constants.DATA_SAVE_PATH, f'{task_name}_errors_T{constants.TIME_NOW}_{process_i}.json'
                )
            if os.path.exists(thread_errors_path):
                with open(thread_errors_path, 'r') as f:
                    thread_errors = json.load(f)
                print(f'thread {process_i}:')
                for k in thread_errors.keys():
                    print(f'{k}         \t\t\t\t{thread_errors[k]}')
                    if k not in merge_errors[task_name].keys():
                        merge_errors[task_name][k] = 0
                    merge_errors[task_name][k] += thread_errors[k] 

    with open(output_errors_path, 'w') as f:
        json.dump(merge_errors, f)
    print('Saved error logs to :', output_errors_path)


def count_successes_from_disk(task_tuple):

    task_name = make_task_name(task_tuple)
    print(task_name)

    success_count = 0
    # top constants.DATA_SAVE_PATH
    # e.g. /data_alfred/sampled/new_trajectories_T20200615_203135_456358
    for root, dirs, files in os.walk(constants.DATA_SAVE_PATH):
        for d in dirs:
            if d.count('-') == 4 and d == task_name:
                for _root, _dirs, _files in os.walk(os.path.join(constants.DATA_SAVE_PATH, d)):
                    for _d in _dirs:
                        for __root, __dirs, __files in os.walk(os.path.join(constants.DATA_SAVE_PATH, d, _d)):
                            if 'video.mp4' in __files:
                                success_count += 1
                break
                        
    return success_count

def sample_task_trajs(
    args, task_tuple, agent, env, obj_to_scene_ids, 
    scene_id_to_objs, pickup_candidates, add_requirements={}):
    '''
    Sample trajectory according to task tuple, save to disk location.

    task_spec:  tuple(  str(goal_type), str(pickup object type), 
                        str(movable receptacle object type),str(receptacle object type), int(scene number)
                    ). 
                Example: ('pick_two_obj_and_place', 'Watch', 'None', 'Dresser', 205)

    add_requirements: optional dict. Example: {'obj_repeat':3, 'seed':42}
    '''

    print("Force Unsave Success Data: %s" % str(args.force_unsave_successes))

    # scene_num is an integer
    gtype, pickup_obj, movable_obj, receptacle_obj, scene_num = task_tuple
    print(f'Task: {task_tuple}')

    # success and failure book-keeping
    # k=error type, v=int count
    errors = {}
    # k=seed, v=traj dir path
    sampled_traj_dirs = {'successes':{}, 'fails':{}}

    # try multiple times
    target_remaining = args.repeats_per_cond - count_successes_from_disk(task_tuple)
    tries_remaining = args.trials_before_fail
    num_place_fails = 0

    # set random seeds -- determines room object locations and agent start pos
    if 'seeds' in add_requirements.keys():
        # all trials wil generate the same results!
        seeds = add_requirements['seeds']
    else:
        seeds = [random.randint(0, 2 ** 32) for _ in range(args.trials_before_fail)]
    print('SEED = ', seeds)

    # optionally specify how 'crowded' the room is with non-task objects
    if 'obj_repeat' in add_requirements.keys():
        obj_repeat = add_requirements['obj_repeat']
    else:
        obj_repeat = None

    while tries_remaining > 0 :

        constants.pddl_goal_type = gtype
        print("PDDLGoalType: " + constants.pddl_goal_type)            

        # determines room agent start pos
        seed = seeds.pop()
        # e.g. 'pick_two_obj_and_place-Watch-None-Dresser-205'
        task_name = make_task_name(task_tuple)
        # create task directory to store plan, trajectory json and raw images
        # task_id e.g. trial_T20200609_122157_214995
        task_id = create_dirs(task_name, seed, obj_repeat)
        # e.g. /data_alfred/sampled/new_trajectories_T.../pick_two_obj_and_place-Watch-None-Dresser-301/trial_T20200609_122157_214995
        task_root = constants.save_path.replace('raw_images/', '')

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
            for obj_type in scene_id_to_objs[str(scene_num)]:
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
                constraint_objs['empty'] = [
                    (r.replace('Basin', ''), num_place_fails * constants.RECEPTACLE_EMPTY_POINTS) 
                    for r in goal_to_invalid_receptacle[gtype]
                ]

            # turn off the lights if task needs it
            constraint_objs['seton'] = []
            if gtype == 'look_at_obj_in_light':
                constraint_objs['seton'].append((receptacle_obj, False))

            # alert user that scene is now sparser if last try failed
            if num_place_fails > 0:
                print("Failed %d placements in the past; increased free point constraints: " % num_place_fails
                        + str(constraint_objs))           

            # thor env spawn up the scene according to constraint objs
            scene_info = {'scene_num': int(scene_num), 'random_seed': seed}
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
            env.reset(int(scene_num))
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
                agent.step(action_dict)
                reward, terminal = agent.get_reward()

            # dump constants.data_dict to file
            dump_data_dict()
            # save images in images_path to video_path
            save_video()
            
            # book keeping
            sampled_traj_dirs['successes'][seed] = task_root

        except Exception as e:

            # report error in stdout
            import traceback
            traceback.print_exc()

            sampled_traj_dirs['fails'][seed] = task_root

            dump_data_dict()

            if str(e) == "API Action Failed: No valid positions to place object found":
                num_place_fails += 1 

            estr = str(e)
            if len(estr) > 120:
                estr = estr[:120]
            if estr not in errors:
                errors[estr] = 0
            errors[estr] += 1
            print("%%%%%%%%%%")
            es = sum([errors[er] for er in errors])
            print("\terrors (%d):" % es)
            for er, v in sorted(errors.items(), key=lambda kv: kv[1], reverse=True):
                if v / es < 0.01:  # stop showing below 1% of errors.
                    break
                print("\t(%.2f) (%d)\t%s" % (v / es, v, er))
            print("%%%%%%%%%%")

        finally:
            tries_remaining -= 1
            target_remaining = args.repeats_per_cond - count_successes_from_disk(task_tuple)

        # optionally delete directory for successful tasks.   
        if args.force_unsave_successes:
            delete_save(args.in_parallel)        

    print("---------------End of Sampling----------------")
    print((gtype, pickup_obj, movable_obj, receptacle_obj, str(scene_num)))
    print(
        'Finished a maximum of {} trials, with {} fails.'.format(
            args.trials_before_fail, sum([errors[er] for er in errors])
        )
    )

    # if this combination resulted in a certain number of failures with no successes, flag it as not possible.
    if tries_remaining == 0 and target_remaining == args.repeats_per_cond:
         print('The specified tuple is too hard to sample any successful trajectories.')

    return sampled_traj_dirs, errors


def main(args, task_names_dict, process_i=0):

    if args.seed is not None:
        # keep agent start pos always the same
        # keep the way object varies across trials predictable
        np.random.seed(args.seed)

    # ---------------------Setup Scene and Object Candidates------------------------
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

    # scene-goal database
    for g in constants.GOALS:
        for st in constants.GOALS_VALID[g]:
            scenes_for_goal[g].extend([str(s) for s in constants.SCENE_TYPE[st]])
        scenes_for_goal[g] = set(scenes_for_goal[g])

    # scene-type database
    for st in constants.SCENE_TYPE:
        for s in constants.SCENE_TYPE[st]:
            scene_to_type[str(s)] = st

    # Union objects that can be placed.
    pickup_candidates = list(
        set().union(*[constants.VAL_RECEPTACLE_OBJECTS[obj] for obj in constants.VAL_RECEPTACLE_OBJECTS])
    )
    pickup_candidates = [p for p in pickup_candidates if constants.OBJ_PARENTS[p] in obj_to_scene_ids]

    # objects that can be used as receptacle
    receptacle_candidates = [obj for obj in constants.VAL_RECEPTACLE_OBJECTS
                             if obj not in constants.MOVABLE_RECEPTACLES and obj in obj_to_scene_ids] + \
                            [obj for obj in constants.VAL_ACTION_OBJECTS["Toggleable"]
                             if obj in obj_to_scene_ids]

    # toaster isn't interesting in terms of producing linguistic diversity
    receptacle_candidates.remove('Toaster')
    receptacle_candidates.sort() 

    # ---------------------Setup Env and Agent--------------------------------------
    # create env and agent
    env = ThorEnv()
    game_state = TaskGameStateFullKnowledge(env)
    agent = DeterministicPlannerAgent(process_id=0, game_state=game_state)

    # ---------------------Sample the tasks one by one------------------------------
    dont_sample_scenes = VALID_UNSEEN_SCENES | TEST_SEEN_SCENES | TEST_UNSEEN_SCENES 

    for curr_task_name in progressbar.progressbar(task_names_dict[process_i]): 

        # construct task tuple
        task_tuple = curr_task_name.split('-')

        # filter out unseen and test environments
        if task_tuple[-1] in dont_sample_scenes:
            continue 

        # call sample_task_trajs
        sampled_traj_dirs, errors = sample_task_trajs(
            args, task_tuple, agent, env, obj_to_scene_ids, scene_id_to_objs, pickup_candidates, {})

        # save the directory paths for success and failed trajectories, 
        # and error counts to disk
        save_bookkeeping_and_splits(
            curr_task_name, constants.DATA_SAVE_PATH, args.splits_dir,
            sampled_traj_dirs, errors, process_i)

def parallel_main(args):
    procs = [
        mp.Process(target=main, args=(args, task_names_dict, process_i)) for process_i in range(args.num_processes)
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
    parser = argparse.ArgumentParser()

    # save settings
    parser.add_argument(
        '--force_unsave_successes', action='store_true', 
        help="don't save any data for successful traj (for debugging purposes)"
    )
    parser.add_argument(
        '--save_path', type=str, help="where to save the success & failure trajectories and data",  
        default='/root/data_alfred/sampled/new_trajectories/'
    )
    parser.add_argument(
        '--splits_dir', type=str, help="where to save the split file",  
        default='/root/data_alfred/splits/'
    )

    # debugging settings
    parser.add_argument(
        '--debug', action='store_true', 
        help="print agent env actions info per timestep."
    )

    parser.add_argument(
        '--x_display', type=str, required=False, default=constants.X_DISPLAY, help="x_display id"
    )

    # multi-process settings
    parser.add_argument(
        "--in_parallel", action='store_true', 
        help="this collection will run in parallel with others, so load from disk on every new sample"
    )
    parser.add_argument(
        "-n", "--num_processes", type=int, default=0, 
        help="number of processes for parallel mode"
    )

    # task params from list of previously failed trajectories
    parser.add_argument(
        '--task_names_path', type=str, 
        help="path to text file. each line is a task name e.g. look_at_obj_in_light-BaseballBat-None-DeskLamp-301"
    )

    # number of tries
    parser.add_argument(
        "--repeats_per_cond", type=int, 
        help='number of successful trajectories we want to collect for each task tuple.', 
        default=3
    )
    parser.add_argument(
        "--trials_before_fail", type=int, 
        help='number of tries before we stop trying for the task tuple. (some task tuples can be very challenging.)', 
        default=5
    )

    parse_args = parser.parse_args()

    parse_args.seed = None

    # each trajectory directory is time stamped by the moment we start collecting them
    constants.TIME_NOW = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    parse_args.save_path = '/'.join(parse_args.save_path.split('/'))
    constants.DATA_SAVE_PATH = parse_args.save_path[:-1] + f'_T{constants.TIME_NOW}/'

    start_time = time.time()

    # get task_names from file
    with open(parse_args.task_names_path, 'r') as f:
        task_names_list = f.read().splitlines()

    task_names_dict = {}
    if parse_args.in_parallel and parse_args.num_processes > 1:
        # divide task among processes
        # TODO: should replace with a proper queue for multi-process
        quotient = len(task_names_list) // parse_args.num_processes
        for process_i in range(parse_args.num_processes):
            task_names_dict[process_i] = task_names_list[process_i*quotient: (process_i+1)*quotient]
            if process_i == parse_args.num_processes-1:
                task_names_dict[process_i] += task_names_list[(process_i+1)*quotient:]

        parallel_main(parse_args)
        merge_process_results(parse_args)
    else:

        task_names_dict[0] = task_names_list
        main(parse_args, task_names_dict)

    print('-------------------------------------------------------')
    print("total time:")
    print(time.time() - start_time)


# Usage
# python scripts/sample_augmentation_trajectories.py --task_names_path scripts/task_names_toy.txt --save_path /data_alfred/sampled/new_trajectories/ --splits ~/data_alfred/splits/ --trials_before_fail 2 --in_parallel --num_processes 3
