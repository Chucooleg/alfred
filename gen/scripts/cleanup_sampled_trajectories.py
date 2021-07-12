'''Clean up sampled partial trajectories. Should run sample_augmentation_trajectories.py first'''

import os
import json
import argparse
import progressbar
import numpy as np
import shutil
from collections import defaultdict


def print_num_subgoal_distribution(task_to_traj):
    
    subgoal_counts = defaultdict(int)

    for k in task_to_traj.keys():
        subgoal_counts[task_to_traj[k]['max_collected_subgoals']] += 1

    print('Distribution: Number of subgoal per sampled trajectory.')
    print(
        sorted(
            [(num_subgoal, ct)for num_subgoal, ct in subgoal_counts.items()], key=lambda x: x[1], 
            reverse=True
        )
    )

def remove_failed_last_step(task_to_traj, data, dout, dout_split, new_split_filename):
    '''
    clean up the collected trajectory to remove the failed last step of any
    partial(i.e. not fully complete) trajectories.
    '''

    split_entries = []
    images_to_move = {}

    for task_name in task_to_traj.keys():

        # e.g. pick_two_obj_and_place-Bread-None-Microwave-20/trial_T20200817_133657_955225
        task_name_trial_name = '/'.join(task_to_traj[task_name]['best_trial_dir'].split('/')[-2:])
        traj_data_out_dir = os.path.join(dout, task_name_trial_name)

        if not os.path.exists(os.path.join(traj_data_out_dir, 'traj_data.json')):
            if not os.path.exists(traj_data_out_dir):
                os.makedirs(traj_data_out_dir)
            
            num_complete_subgoals = task_to_traj[task_name]['max_collected_subgoals'] 
            traj_data_p = os.path.join(data, task_to_traj[task_name]['best_trial_dir'], 'traj_data.json')
            with open(traj_data_p, 'r') as f:
                traj_data = json.load(f)

            if not task_to_traj[task_name]['full_traj_success']:
                # remove the failed last subgoal from plan
                traj_data['plan']['high_pddl'].pop()
                # e.g. 5 complete, high idx to keep 0, 1, 2, 3, 4, all low actions with high idx 5 should be removed
                while traj_data['plan']['low_actions'][-1]['high_idx'] >= num_complete_subgoals:
                    traj_data['plan']['low_actions'].pop()
                
                # remove the failed last subgoal from language template
                traj_data['template']['high_descs'].pop()

                # remove the failed last subgoal from image frames
                while traj_data['images'][-1]['high_idx'] >= num_complete_subgoals:
                    traj_data['images'].pop()

            # prepare to copy the only useful images over
            images_to_move[task_name] = [
                (os.path.join(data, task_to_traj[task_name]['best_trial_dir'], 'raw_images', image_entry['image_name']),
                os.path.join(traj_data_out_dir, 'raw_images', image_entry['image_name']))
                for image_entry in traj_data['images']]

            # save the trajectory to dout.
            traj_data_out_p = os.path.join(traj_data_out_dir, 'traj_data.json')
            with open(traj_data_out_p, 'w') as f:
                json.dump(traj_data, f)

            # Save to raw split
            split_entries.append({
                'task':task_name_trial_name, 
                'repeat_idx':0, 
                'full_traj_success':task_to_traj[task_name]['full_traj_success'],
                'collected_subgoals': task_to_traj[task_name]['collected_subgoals']
            })

    # Write raw split
    # e.g. /data_alfred/splits/sample_failed_20200820_raw.json
    split_path = os.path.join(dout_split, new_split_filename)
    with open(split_path, 'w') as f:
        json.dump({'augmentation':split_entries}, f)
    print(f'Successfully wrote new split to path {split_path}')

    return images_to_move

def filter_trajectories(task_name_list, data_dir, sampl_dir):
    '''
    filter down to sampled trajectories with 
    1) at least 1 successful subgoal and 
    2) given a task name, the best trial with max number of subgoal
    '''

    task_to_traj = {}
    missing = []

    print('Start filtering raw sampled trajectories.')
    for task_name in progressbar.progressbar(task_name_list):

        task_info = {'best_trial_dir':None, 'max_collected_subgoals':0, 'full_traj_success':False}
        found_task_dir = False
        task_dir = os.path.join(data_dir, sampl_dir, task_name)

        if os.path.exists(task_dir):
            found_task_dir = True
            for trial_dir in os.listdir(task_dir):
                traj_data_p = os.path.join(task_dir, trial_dir, 'traj_data.json')
                if os.path.exists(traj_data_p):
                    with open(traj_data_p, 'r') as f:
                        traj_data = json.load(f)
                    collected_num_subgoals = len(traj_data['plan']['high_pddl'])
                    if os.path.exists(os.path.join(task_dir, trial_dir, 'video.mp4')):
                        # save successful full trajectory
                        task_info['best_trial_dir'] = os.path.join(sampl_dir, task_name, trial_dir)
                        task_info['max_collected_subgoals'] = collected_num_subgoals
                        task_info['full_traj_success'] = True
                        task_to_traj[task_name] = task_info
                        break
                    else:
                        # save the longest version. save only if at least one subgoal has been completed. 
                        if len(
                            traj_data['plan']['high_pddl']) > task_info['collected_subgoals'] and \
                                len(traj_data['plan']['high_pddl']
                            ) > 1:
                            # the last subgoal failed, so remove it from the count
                            task_info['collected_subgoals'] = collected_num_subgoals - 1
                            task_info['best_trial_dir'] = os.path.join(sampl_dir, task_name, trial_dir)
                            task_info['full_traj_success'] = False
                            task_to_traj[task_name] = task_info        

        if not found_task_dir:
            missing.append(task_name)  

    print(f'Finished filtering raw sampled trajectories. {len(missing)} tasks are missing.')
    return task_to_traj, missing
 
def move_images(images_to_move):
    '''Only copy useful images from sampling directory to dout'''
    dest_paths = []
    print('Copying images from sampling directory to dout')
    for key in progressbar.progressbar(images_to_move.keys()):
        for src_fpath, dest_fpath in images_to_move[key]:
            os.makedirs(os.path.dirname(dest_fpath), exist_ok=True)
            dest_paths.append(shutil.copy(src_fpath, dest_fpath))
    print(f'Finished copying {len(dest_paths)} images from sampling directory to dout')

def main(parse_args):
    
    # load tasks we tried to sample for.
    with open(parse_args.task_names_path, 'r') as f:
        task_name_list = f.read().splitlines()
        print(f'Loaded {len(task_name_list)} task names.')

    # filter down to sampled trajectories with 
    # 1) at least 1 successful subgoal and 
    # 2) given a task name, the best trial with max number of subgoal
    task_to_traj, missing = filter_trajectories(
        task_name_list, parse_args.data_dir, parse_args.sampl_dir
    )

    print_num_subgoal_distribution(task_to_traj)

    images_to_move = remove_failed_last_step(
        task_to_traj, parse_args.data_dir, parse_args.dout, parse_args.dout_split
    )
    move_images(images_to_move)

    print('Finished cleaning up sampled trajectories and saved to dout.')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data_dir', type=str,
        help="where the new trajectories are saved. \
              e.g. /data_alfred/sampled/"
    )
    parser.add_argument(
        '--sampl_dir', type=str,
        help="where the new trajectories are saved. \
              e.g. new_trajectories_T..."
    )
    parser.add_argument(
        '--dout', type=str,
        help="directory to write cleaned data to. \
              e.g. /data_alfred/json_data_augmentation_20200820/"
    )
    parser.add_argument(
        '--dout_split', type=str,
        help="directory to write new split to. \
              e.g. /data_alfred/splits/"
    )
    parser.add_argument(
        '--new_split_filename', type=str,
        help="new split filename. \
              e.g. sample_failed_20200820_raw.json"
    )
    parser.add_argument(
        '--task_names_path', type=str,
        help="path to text file containing previously failed tasks.  \
            Each line is a task name \
            e.g. look_at_obj_in_light-BaseballBat-None-DeskLamp-301. \
            e.g. gen/scripts/task_names_toy.txt"
    )

    parse_args = parser.parse_args()

    main(parse_args)
