import json
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

def collect_task(split_dir):
    task_list = []
    split_dir = split_dir.rstrip(os.path.sep)
    assert os.path.isdir(split_dir)
    num_sep = split_dir.count(os.path.sep)
    for root, dirs, files in os.walk(split_dir):
        if 'raw_images' in root:
            task = root.replace(split_dir + '/', '').rstrip('/raw_images')
            assert '/trial_T' in task
            task_list.append({
                'repeat_idx': 0,
                'task': task
            })
    return task_list


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    # settings
    parser.add_argument('--data_split_dir', help='data dir at split level. e.g. data/json_feat_2.1.0/seen/')
    parser.add_argument('--split_name', help='train, valid_seen, valid_unseen, test_seen, or test_unseen')
    parser.add_argument('--output_split_dir', help='paht to store output split json file. e.g. data/splits/jan01.json')

    # parser
    args = parser.parse_args()
    
    # collect tasks
    # {'train':
    #     [{'repeat_idx': 0,
    #       'task': 'pick_cool_then_place_in_recep-LettuceSliced-None-DiningTable-17/trial_T20190909_070538_437648'},
    #      {'repeat_idx': 1,
    #       'task': 'pick_cool_then_place_in_recep-LettuceSliced-None-DiningTable-17/trial_T20190909_070538_437648'},
    #      ...
    #     ]
    # }
    out_split = {args.split_name: collect_task(args.data_split_dir)}

    # save file out
    with open(args.output_split_dir, 'w') as f:
        json.dump(out_split, f)