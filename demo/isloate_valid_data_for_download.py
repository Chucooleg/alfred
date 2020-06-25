# sudo cp -r json_feat_2.1.0/valid_unseen/ json_demo_cache/
# sudo cp -r json_feat_2.1.0/valid_seen/ json_demo_cache/

import json
import os
from collections import defaultdict
import shutil
import progressbar

with open('/root/data_alfred/splits/collect_states_20200511_valid_seen_notebook_success_paths.json', 'r') as f:
    valid_seen_object_state_collection_success_paths = json.load(f)
valid_seen_paths = [p.split('/')[4:6] for p in valid_seen_object_state_collection_success_paths]


with open('/root/data_alfred/splits/collect_states_20200511_valid_unseen_notebook_success_paths.json', 'r') as f:
    valid_unseen_object_state_collection_success_paths = json.load(f)
valid_unseen_paths = [p.split('/')[4:6] for p in valid_unseen_object_state_collection_success_paths]

old_root_base = '/root/data_alfred/json_feat_2.1.0/'
new_root_base = '/root/data_alfred/json_demo_cache/'

for paths in [valid_seen_paths, valid_unseen_paths]:
    for path in progressbar.progressbar(paths):
        for suffix in ['pp_model:seq2seq_nl_with_frames,name:v1.5_epoch_50_high_level_instrs', 'pp_model:seq2seq_per_subgoal,name:v2_epoch_40_obj_instance_enc_max_pool_dec_aux_loss_weighted_bce_1to2']:
            
            # /data_alfred/json_feat_2.1.0/look_at_obj_in_light-CellPhone-None-FloorLamp-219/trial_T20190908_044113_026049/pp_model:seq2seq_per_subgoal,name:v2_epoch_40_obj_instance_enc_max_pool_dec_aux_loss_weighted_bce_1to2
            old_dir_path = os.path.join(old_root_base, '/'.join(path), suffix)
            # /data_alfred/json_demo_cache/look_at_obj_in_light-CellPhone-None-FloorLamp-219/trial_T20190908_044113_026049
            new_dir_path = os.path.join(new_root_base, '/'.join(path), suffix)

            try:
                assert os.path.exists(old_dir_path)
            except:
                import pdb; pdb.set_trace()

            # if not os.path.exists(new_dir_path):
            #     os.makedirs(new_dir_path)
            
            dest = shutil.copytree(old_dir_path, new_dir_path)
