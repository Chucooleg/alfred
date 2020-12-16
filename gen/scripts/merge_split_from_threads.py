import os
import json
from argparse import ArgumentParser
from collections import defaultdict


def merge_thread_splits(thread_paths_to_merge):
    merged_split = defaultdict(list)
    ct = 0
    for thread_p in thread_paths_to_merge:
        with open(thread_p, 'r') as f:
            thread_split = json.load(f)
            ct += len(thread_split['augmentation'])
        for split_k in thread_split:
            merged_split[split_k] += thread_split[split_k]

    return merged_split


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--raw_splits', help='json file containing raw splits coming directly out from planner.', default='/root/data_alfred/splits/unlabeled_12k_20201206_raw.json')
    parse_args = parser.parse_args()

    # unlabeled_12k_20201206_raw.json
    raw_split = parse_args.raw_splits.split('/')[-1]
    # unlabeled_12k_20201206_
    split_prefix = '_'.join(raw_split.split('_')[:-1]) + '_'
    # /root/data_alfred/splits/
    raw_split_dir = parse_args.raw_splits.rstrip(raw_split)

    failed_split_paths = []
    success_split_paths = []
    for root, dirs, files in os.walk(raw_split_dir):
        for f in files:
            if split_prefix in f and 'thread' in f and not 'raw' in f:
                if 'failed' in f:
                    failed_split_paths.append(os.path.join(root, f))
                else:
                    success_split_paths.append(os.path.join(root, f))

    merged_failed_split = merge_thread_splits(failed_split_paths)
    merged_success_split = merge_thread_splits(success_split_paths)

    out_success_split_path = os.path.join(raw_split_dir, split_prefix.rstrip('_') + '.json')
    out_failed_split_path = os.path.join(raw_split_dir, split_prefix.rstrip('_') + '_failed.json')
    
    with open(out_success_split_path, 'w') as f:
        json.dump(merged_success_split, f)
        print(f'Saved merged success split to {out_success_split_path}')

    with open(out_failed_split_path, 'w') as f:
        json.dump(merged_failed_split, f)
        print(f'Saved failed failed split to {out_failed_split_path}')
