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


def print_successes(succ_traj):
    print("###################################\n")
    print("Successes: ")
    print(succ_traj)
    print("\n##################################")


def main(args, sampled_task=('pick_two_obj_and_place', 'Watch', 'None', 'Dresser', 205), obj_repeats_var=3, seed_var=42):
    # settings
    constants.DATA_SAVE_PATH = args.save_path
    print("Force Unsave Data: %s" % str(args.force_unsave))

    # # Set up data structure to track dataset balance and use for selecting next parameters.
    # # In actively gathering data, we will try to maximize entropy for each (e.g., uniform spread of goals,
    # # uniform spread over patient objects, uniform recipient objects, and uniform scenes).
    # succ_traj = pd.DataFrame(columns=["goal", "pickup", "movable", "receptacle", "scene"])

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
    # for g in constants.GOALS:
    #     for st in constants.GOALS_VALID[g]:
    #         scenes_for_goal[g].extend([str(s) for s in constants.SCENE_TYPE[st]])
    #     scenes_for_goal[g] = set(scenes_for_goal[g])

    # scene-type database
    # for st in constants.SCENE_TYPE:
    #     for s in constants.SCENE_TYPE[st]:
    #         scene_to_type[str(s)] = st

    # pre-populate counts in this structure using saved trajectories path.
    # succ_traj, full_traj = load_successes_from_disk(args.save_path, succ_traj, args.just_examine, args.repeats_per_cond)
    # if args.just_examine:
    #     print_successes(succ_traj)
    #     return

    # pre-populate failed trajectories.
    # fail_traj = load_fails_from_disk(args.save_path)
    # print("Loaded %d known failed tuples" % len(fail_traj))

    # create env and agent
    env = ThorEnv()

    game_state = TaskGameStateFullKnowledge(env)
    agent = DeterministicPlannerAgent(thread_id=0, game_state=game_state)

    errors = {}  # map from error strings to counts, to be shown after every failure.
    # goal_candidates = constants.GOALS[:]
    pickup_candidates = list(set().union(*[constants.VAL_RECEPTACLE_OBJECTS[obj]  # Union objects that can be placed.
                                           for obj in constants.VAL_RECEPTACLE_OBJECTS]))
    # ['Apple', 'AppleSliced', 'ButterKnife', 'Egg', ...]
    pickup_candidates = [p for p in pickup_candidates if constants.OBJ_PARENTS[p] in obj_to_scene_ids]
    # movable_candidates = list(set(constants.MOVABLE_RECEPTACLES).intersection(obj_to_scene_ids.keys()))
    # receptacle_candidates = [obj for obj in constants.VAL_RECEPTACLE_OBJECTS
    #                          if obj not in constants.MOVABLE_RECEPTACLES and obj in obj_to_scene_ids] + \
    #                         [obj for obj in constants.VAL_ACTION_OBJECTS["Toggleable"]
    #                          if obj in obj_to_scene_ids]

    # toaster isn't interesting in terms of producing linguistic diversity
    # receptacle_candidates.remove('Toaster')
    # receptacle_candidates.sort()

    # scene_candidates = list(scene_id_to_objs.keys())

    # n_until_load_successes = args.async_load_every_n_samples  ## only when parallel
    # print_successes(succ_traj)
    # task_sampler = sample_task_params(succ_traj, full_traj, fail_traj,
    #                                   goal_candidates, pickup_candidates, movable_candidates,
    #                                   receptacle_candidates, scene_candidates)

    # NOTE replaced sampled_task here with arguments
    # sampled_task = (g, p, m, r, int(s))
    # gtype, pickup_obj, movable_obj, receptacle_obj, sampled_scene
    # sampled_task = ('pick_two_obj_and_place', 'Watch', 'None', 'Dresser', 205)

    # keeps trying out new task tuples as trajectories either fail or suceed
    print(sampled_task)  # DEBUG
    # if sampled_task is None:
    #     sys.exit("No valid tuples left to sample (all are known to fail or already have %d trajectories" %
    #                 args.repeats_per_cond)
    gtype, pickup_obj, movable_obj, receptacle_obj, sampled_scene = sampled_task
    print("sampled tuple: " + str((gtype, pickup_obj, movable_obj, receptacle_obj, sampled_scene)))

    tries_remaining = args.trials_before_fail
    # only try to get the number of trajectories left to make this tuple full.
    # target_remaining = args.repeats_per_cond - len(succ_traj.loc[(succ_traj['goal'] == gtype) &
    #                                                         (succ_traj['pickup'] == pickup_obj) &
    #                                                         (succ_traj['movable'] == movable_obj) &
    #                                                         (succ_traj['receptacle'] == receptacle_obj) &
    #                                                         (succ_traj['scene'] == str(sampled_scene))])
    num_place_fails = 0  # count of errors related to placement failure for no valid positions.
    successful_task_paths = []

    # continue until we're (out of tries + have never succeeded) or (have gathered the target number of instances)
    # while tries_remaining > 0 and target_remaining > 0:
    while tries_remaining > 0: # and target_remaining > 0:

        # environment setup
        constants.pddl_goal_type = gtype
        print("PDDLGoalType: " + constants.pddl_goal_type)
        # create save dir 'dataset/new_trajectories/pick_two_obj_and_place-Watch-None-Dresser-205/trial_T20190907_181954_161870/raw_images/
        task_id, task_name, task_path = create_dirs(gtype, pickup_obj, movable_obj, receptacle_obj, sampled_scene, obj_repeats_var, seed_var)

        # setup data dictionary
        setup_data_dict()
        constants.data_dict['task_id'] = task_id
        constants.data_dict['task_type'] = constants.pddl_goal_type
        constants.data_dict['dataset_params']['video_frame_rate'] = constants.VIDEO_FRAME_RATE

        # plan & execute
        try:
            # Agent reset to new scene.

            # 1. How many of the pickup objects to spawn 
            # 2. How sparse should the receptacle objects be
            constraint_objs = {'repeat': [(constants.OBJ_PARENTS[pickup_obj],  # Generate multiple parent objs.
                                            np.random.randint(2 if gtype == "pick_two_obj_and_place" else 1,
                                                                constants.PICKUP_REPEAT_MAX + 1))],
                                'sparse': [(receptacle_obj.replace('Basin', ''),
                                            num_place_fails * constants.RECEPTACLE_SPARSE_POINTS)]}
            
            # 3. How many of the required movable receptacle type to spawn
            if movable_obj != "None":
                constraint_objs['repeat'].append((movable_obj,
                                                    np.random.randint(1, constants.PICKUP_REPEAT_MAX + 1)))

            # 4. Spawn a bunch of pickupable, non-target pickup object or movable receptacles
            # constrained to only objects listed in scene layout
            for obj_type in scene_id_to_objs[str(sampled_scene)]:
            # for obj_type in scene_id_to_objs[str(29)]:
                if (obj_type in pickup_candidates and
                        obj_type != constants.OBJ_PARENTS[pickup_obj] and obj_type != movable_obj):
                    # constraint_objs['repeat'].append((obj_type,
                    #                                     np.random.randint(1, constants.MAX_NUM_OF_OBJ_INSTANCES + 1)))
                    constraint_objs['repeat'].append((obj_type,
                                                      obj_repeats_var))

            if gtype in goal_to_invalid_receptacle:
                constraint_objs['empty'] = [(r.replace('Basin', ''), num_place_fails * constants.RECEPTACLE_EMPTY_POINTS)
                                            for r in goal_to_invalid_receptacle[gtype]]

            # 6. Turn off the lights
            constraint_objs['seton'] = []
            if gtype == 'look_at_obj_in_light':
                constraint_objs['seton'].append((receptacle_obj, False))


            if num_place_fails > 0:
                print("Failed %d placements in the past; increased free point constraints: " % num_place_fails
                        + str(constraint_objs))

            # Set up the scene
            # read agent_base.py, def reset() 
            # game_state_base.py def reset() , thor environment literally spawn up according to scene number of constraint_objs
            # returns:
            # (dataset_type, task_row), max_num_repeats for object type, remove_prob
            # ('train', '9999'), 3, 0.0       
            # scene_info = {'scene_num': sampled_scene, 'random_seed': random.randint(0, 2 ** 32)}
            scene_info = {'scene_num': sampled_scene, 'random_seed': seed_var}
            info = agent.reset(scene=scene_info,
                                objs=constraint_objs)

            # Problem initialization with given constraints.
            task_objs = {'pickup': pickup_obj}
            if movable_obj != "None":
                task_objs['mrecep'] = movable_obj
            if gtype == "look_at_obj_in_light":
                task_objs['toggle'] = receptacle_obj
            else:
                task_objs['receptacle'] = receptacle_obj

            # task_game_state.py def setup_problem
            # setup constants.data_dict['pddl_params'] for keys 
            # 'object_target' 
            # 'object_sliced'
            # 'parent_target'
            # 'toggle_target'
            # 'mrecep_target'
            agent.setup_problem({'info': info}, scene=scene_info, objs=task_objs)

            # Now that objects are in their initial places, record them. e.g. keep track of their poses.
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

            # Pre-restore the scene to cause objects to "jitter" like they will when the episode is replayed
            # based on stored object and toggle info. This should put objects closer to the final positions they'll
            # be inlay at inference time (e.g., mugs fallen and broken, knives fallen over, etc.).
            print("Performing reset via thor_env API")
            env.reset(sampled_scene)
            print("Performing restore via thor_env API")
            env.restore_scene(object_poses, object_toggles, dirty_and_empty)
            event = env.step(dict(constants.data_dict['scene']['init_action']))

            terminal = False
            while not terminal and agent.current_frame_count <= constants.MAX_EPISODE_LENGTH:
                action_dict = agent.get_action(None)
                # deterministic_planner_agent.py def step()
                # semantic_map_planner_agent.py def step()
                # plan_agent.py def execute_plan()
                # plan = self.game_state.get_current_plan(force_update=True)
                # planned_game_state.py def get_current_plan()
                # self.plan = self.planner.get_plan()
                # ff_planner_handler.PlanParser(domain_path)
                # ff_planner_handler.py PlanParser def get_plan()
                # ff_planner_handler.py def get_plan_async()
                # ln 135 def get_plan_from_file() call solver!
                agent.step(action_dict)
                reward, terminal = agent.get_reward()

            # dump constants.data_dict to file
            dump_data_dict()
            # save images in images_path to video_path
            save_video()

            successful_task_paths.append(task_path)
            tries_remaining -= 1

        except Exception as e:
            import traceback
            traceback.print_exc()
            print("Error: " + repr(e))
            print("Invalid Task: skipping...")
            if args.debug:
                print(traceback.format_exc())

        # deleted = delete_save(args.in_parallel)
            # deleted = delete_save(True)
            # if not deleted:  # another thread is filling this task successfully, so leave it alone.
            #     target_remaining = 0  # stop trying to do this task.
            # else:
            #     if str(e) == "API Action Failed: No valid positions to place object found":
            #         # Try increasing the space available on sparse and empty flagged objects.
            #         num_place_fails += 1
            #         tries_remaining -= 1
            #     else:  # generic error
            #         tries_remaining -= 1

            if str(e) == "API Action Failed: No valid positions to place object found":
                # Try increasing the space available on sparse and empty flagged objects.
                num_place_fails += 1
                tries_remaining -= 1
            else:  # generic error
                tries_remaining -= 1

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

            # for debugging
            # delete_save(False)
            save_video()

            continue

        # if args.force_unsave:
        #     # delete_save(args.in_parallel)
        #     delete_save(False)


        # add to save structure.
        # succ_traj = succ_traj.append({
        #     "goal": gtype,
        #     "movable": movable_obj,
        #     "pickup": pickup_obj,
        #     "receptacle": receptacle_obj,
        #     "scene": str(sampled_scene)}, ignore_index=True)
        # target_remaining -= 1
        # tries_remaining += args.trials_before_fail  # on success, add more tries for future successes

    # if tries_remaining == 0 and target_remaining == args.repeats_per_cond:
    if tries_remaining == 0 :
        print('Finished {} trials, with {} fails.'.format(args.trials_before_fail, num_place_fails))
        print((gtype, pickup_obj, movable_obj, receptacle_obj, str(sampled_scene)))
        print('successful paths:')
        for p in successful_task_paths:
            print(p)

    return task_name, successful_task_paths

    # if this combination resulted in a certain number of failures with no successes, flag it as not possible.
    # if tries_remaining == 0 and target_remaining == args.repeats_per_cond:
    #     new_fails = [(gtype, pickup_obj, movable_obj, receptacle_obj, str(sampled_scene))]
    #     fail_traj = load_fails_from_disk(args.save_path, to_write=new_fails)
    #     print("%%%%%%%%%%")
    #     print("failures (%d)" % len(fail_traj))
    #     # print("\t" + "\n\t".join([str(ft) for ft in fail_traj]))
    #     print("%%%%%%%%%%")

    # # if this combination gave us the repeats we wanted, note it as filled.
    # if target_remaining == 0:
    #     full_traj.add((gtype, pickup_obj, movable_obj, receptacle_obj, sampled_scene))

    # if we're sharing with other processes, reload successes from disk to update local copy with others' additions.
    # if args.in_parallel:
    #     if n_until_load_successes > 0:
    #         n_until_load_successes -= 1
    #     else:
    #         print("Reloading trajectories from disk because of parallel processes...")
    #         succ_traj = pd.DataFrame(columns=succ_traj.columns)  # Drop all rows.
    #         succ_traj, full_traj = load_successes_from_disk(args.save_path, succ_traj, False, args.repeats_per_cond)
    #         print("... Loaded %d trajectories" % len(succ_traj.index))
    #         n_until_load_successes = args.async_load_every_n_samples
    #         print_successes(succ_traj)
    #         task_sampler = sample_task_params(succ_traj, full_traj, fail_traj,
    #                                             goal_candidates, pickup_candidates, movable_candidates,
    #                                             receptacle_candidates, scene_candidates)
    #         print("... Created fresh instance of sample_task_params generator")


def create_dirs(gtype, pickup_obj, movable_obj, receptacle_obj, scene_num, obj_repeats_var, seed_var):
    task_id = 'trial_T' + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    task_name = '%s-%s-%s-%s-%d' % (gtype, pickup_obj, movable_obj, receptacle_obj, scene_num)
    save_name = task_name + '/' + f'seed_{seed_var}_repeat_{obj_repeats_var}/' + task_id

    constants.save_path = os.path.join(constants.DATA_SAVE_PATH, save_name, RAW_IMAGES_FOLDER)
    if not os.path.exists(constants.save_path):
        os.makedirs(constants.save_path)

    print("Saving images to: " + constants.save_path)
    return task_id, task_name, constants.save_path


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
    save_folder = constants.save_path.replace(RAW_IMAGES_FOLDER, '')
    if os.path.exists(save_folder):
        try:
            shutil.rmtree(save_folder)
        except OSError as e:
            if in_parallel:  # another thread succeeded at this task while this one failed.
                return False
            else:
                raise e  # if we're not running in parallel, this is an actual.
    return True


# def parallel_main(args):
#     procs = [mp.Process(target=main, args=(args,)) for _ in range(args.num_threads)]
#     try:
#         for proc in procs:
#             proc.start()
#             time.sleep(0.1)
#     finally:
#         for proc in procs:
#             proc.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # settings
    parser.add_argument('--force_unsave', action='store_true', help="don't save any data (for debugging purposes)")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--save_path', type=str, default="/root/data_alfred/demo_generated/", help="where to save the generated data")
    parser.add_argument('--x_display', type=str, required=False, default=constants.X_DISPLAY, help="x_display id")
    # parser.add_argument("--just_examine", action='store_true', help="just examine what data is gathered; don't gather more")
    # parser.add_argument("--in_parallel", action='store_true', help="this collection will run in parallel with others, so load from disk on every new sample")
    # parser.add_argument("-n", "--num_threads", type=int, default=0, help="number of processes for parallel mode")
    # parser.add_argument('--json_file', type=str, default="", help="path to json file with trajectory dump")

    # task params
    parser.add_argument("--goal", type=str, help='goal such as pick_two_obj_and_place')
    parser.add_argument("--pickup", type=str, help='object name')
    parser.add_argument("--movable", type=str, help='movable receptacle name')
    parser.add_argument("--receptacle", type=str, help='receptacle name')
    parser.add_argument("--scene", type=int, help='scene number')

    # params
    # parser.add_argument("--repeats_per_cond", type=int, default=3)
    parser.add_argument("--trials_before_fail", type=int, default=5)
    # parser.add_argument("--async_load_every_n_samples", type=int, default=10)

    parse_args = parser.parse_args()

    # if parse_args.in_parallel and parse_args.num_threads > 1:
    #     parallel_main(parse_args)
    # else:
    #     main(parse_args)
    sampled_task = (args.goal, args.pickup, args.movable, args.receptacle, args.scene)

    main(parse_args, sampled_task)


# export ALFRED_ROOT=/root/data/home/hoyeung/alfred/gen/scripts
# python deploy_planner.py --force_unsave --debug --save_path /root/data_alfred/dummy/ --x_display 0