'''
(Pre Explainer/Baseline Instruction Labeling) 
Filter down to only tasks with subgoal lengths match between actions and extracted object states features

(Post Explainer/Baseline Instruction Labeling)
Truncate predicted instruction subgoals to match number of true subgoals
'''

import os
import json
from collections import Counter
import numpy as np
import progressbar
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser


def filter_out_misaligned_action_and_object_states(split, data_dir):

    misaligned_count = 0
    # full traj was collected
    misaligned_success_tasks = []
    # only partial traj was collected 
    misaligned_failed_tasks = []
    error_free_split = []

    for task in progressbar.progressbar(split):

        traj_data_p = os.path.join(data_dir, task['task'], 'traj_data.json')
        with open(traj_data_p, 'r') as f:
            traj_data = json.load(f)
      
        extracted_features_p = os.path.join(
            data_dir, task['task'], 'extracted_feature_states.json'
        )
        with open(extracted_features_p, 'r') as f:
            extracted_features = json.load(f)        

        subgoal_len_features = [(i,len(subgoal)) for i, subgoal in enumerate(extracted_features['instance_visibile'])]
        counter_items = list(Counter([low_a['high_idx'] for low_a in traj_data['plan']['low_actions']]).items())
        counter_items_num = list(Counter(traj_data['num']['low_to_high_idx']).items())   

        if task['full_traj_success']:
            subgoal_len_features = subgoal_len_features[:-1]

        if counter_items == subgoal_len_features:
            error_free_split.append(task)
        else:
            if task['full_traj_success']:
                misaligned_success_tasks.append(task)
            else:
                misaligned_failed_tasks.append(task)

            misaligned_count += 1

    print(f'Found {misaligned_count} misaligned tasks out of {len(split)} tasks.')
    return error_free_split, misaligned_success_tasks, misaligned_failed_tasks

def truncate_extra_subgoals(traj_data, num_subgoals, key):
    assert len(traj_data[key]['anns'][0]['high_descs']) >= num_subgoals
    if len(traj_data[key]['anns'][0]['high_descs']) > num_subgoals:
        traj_data[key]['anns'][0]['high_descs'] = traj_data[key]['anns'][0]['high_descs'][:num_subgoals]
        return True
    else:
        return False

def match_post_prediction_subgoal_lengths(split, data_p, lm_tags, overwrite_traj=False, debug=False):
    '''
    split: a list of tasks {'task':<task name>/<trial id>, 'repeat_idx':int, 'full_traj_success':boolean, 'collected_subgoals':int}
    '''
    adjusts = {k:0 for k in  lm_tags}
    
    for task in split:
        traj_data_p = os.path.join(data_p, task['task'], 'traj_data.json')
        with open(traj_data_p, 'r') as f:
            traj_data = json.load(f)

        if debug:
            print (task)
            for k in lm_tags:
                print (k)
                print (traj_data[k + '_annotations'])
            
        true_num_subgoals = len(traj_data['num']['action_high'])-1
        
        # verify that the predicted instructions has # subgoals >= gold # subgoals
        # when predicting in a batch of different tasks, model can decode more than necessary

        for k in lm_tags:
            assert len(traj_data[k + '_annotations']['anns'][0]['high_descs']) >= true_num_subgoals
        
        for k in lm_tags:
            ann_key = k + '_annotations'
            adjusts[k] += int(truncate_extra_subgoals(traj_data, true_num_subgoals, key=ann_key))

        if debug:
            print (adjusts)
            print (task)
            for k in lm_tags:
                print (k)
                print (traj_data[k + '_annotations'])
                print ('\n\n\n')
        
        if overwrite_traj:
            with open(traj_data_p, 'w') as f:
                json.dump(traj_data, f)
    
    return adjusts

def main(parse_args):

    # load splits
    with open(parse_args.splits, 'r') as f:
        split = json.load(f)['augmentation']
    print('number of trajectories in split = ', len(split))

    if parse_args.pre_auto_labeling:

        # filter
        error_free_split, _, _ = \
            filter_out_misaligned_action_and_object_states(split, parse_args.data)
        # save the filtered split out
        filtered_split_name = parse_args.splits.replace('_filtered.json', '_aligned.json')
        with open(filtered_split_name, 'w') as f:
            json.dump(error_free_split, f)
        print(f'Saved filtered split out to {filtered_split_name}')

    else:

        # match and overwrite trajectory data
        adjusts = match_post_prediction_subgoal_lengths(
                split, parse_args.data, parse_args.lm_tags, overwrite_traj=True, debug=False)
        print('Count adjustments by LM:', adjusts)
        print(f'Finished matching. Overwrote {parse_args.splits}')


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        '--data', help='dataset directory.', type=str, required=True
    )
    parser.add_argument(
        '--splits', help='path to json file containing raw trajectory splits.', type=str, required=True
    )
    parser.add_argument(
        '--pre_auto_labeling', help='filtering pre language model auto-labeling', action='store_true'
    )
    parser.add_argument(
        '--lm_tags', nargs='+', help='tag for all langauge models to be processed. \
            Use like: -lm_tags explainer explainer_auxonly explainer_enconly baseline', 
        required=True
    )

    parse_args = parser.parse_args()
    if parse_args.pre_auto_labeling:
        assert isinstance(parse_args.lm_tags, list) and len(parse_args.lm_tags)

    main(parse_args)