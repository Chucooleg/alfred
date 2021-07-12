'''From collected object states, construct data structures that can be used for training agents.'''

import json
from collections import defaultdict
from vocab import Vocab
import os
import progressbar
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections import Counter


object_state_types = ['isToggled', 'isBroken', 'isFilledWithLiquid', 'isDirty',
                      'isUsedUp', 'isCooked', 'ObjectTemperature', 'isSliced',
                      'isOpen', 'isPickedUp', 'mass', 'parentReceptacles']

receptacle_key = 'receptacleObjectIds'

subgoal_count_specific_features = [
    'objectReceptacleList',
    'objectReceptacleList_TypeNum',
    'instance_visibile',
    'instance_state_change',
    'instance_state_change_since_last_subgoal',
    'instance_distance',
    'instance_receptacle_change',
    'instance_receptacle_change_since_last_subgoal',
    'type_visibile',
    'type_state_change',
    'type_state_change_since_last_subgoal'
]

def shift_states(saved_states):
    '''shift all the states to the last time step for alignment'''
    shifted_states = [{} for _ in range(len(saved_states))]

    for t, state in enumerate(saved_states):
        new_state = {k:v for k,v in state.items() if k != 'objects_metadata'}
        if t < len(saved_states) - 1:
            new_state['objects_metadata'] = saved_states[t+1]['objects_metadata']
        elif t == len(saved_states) - 1: # <<stop>> action
            new_state['objects_metadata'] = saved_states[t]['objects_metadata']
        shifted_states[t] = new_state
        
    return shifted_states

def get_object_states(obj, required_state_types):
    '''get object state that will be used as features'''
    o_states = {}
    for state_typ in required_state_types:
        if state_typ in obj.keys():
            o_states[state_typ] = obj[state_typ]
        else:
            o_states[state_typ] = None
    return o_states

def detect_state_change(last_obj_states, curr_obj_states, objectId_list):
    '''detect object state change from one timestep to the next'''
    state_change = []
    for obj_Id in objectId_list:
        if (not obj_Id in last_obj_states) and (not obj_Id in curr_obj_states):
            state_change.append(False)
        elif not obj_Id in last_obj_states:
            state_change.append(True)
        elif not obj_Id in curr_obj_states:
            raise Exception('Objects should always appear in next time step.')
        else:
            if last_obj_states[obj_Id] == curr_obj_states[obj_Id]:
                state_change.append(False)
            else:
                state_change.append(True)
    assert len(state_change) == len(objectId_list)
    return state_change

def detect_receptacle_change(last_obj_receptacles, curr_obj_receptacles, objectId_list):
    '''detect receptable state change from one timestep to the next'''
    receptacle_change = []
    for obj_Id in objectId_list:
        # TODO
        if (not obj_Id in last_obj_receptacles) and (not obj_Id in curr_obj_receptacles):
            receptacle_change.append(False)
        elif not obj_Id in last_obj_receptacles:
            receptacle_change.append(True)
        elif not obj_Id in curr_obj_receptacles:
            raise Exception('Objects should always appear in next time step.')
        else:
            if last_obj_receptacles[obj_Id] == curr_obj_receptacles[obj_Id]:
                receptacle_change.append(False)
            else:
                receptacle_change.append(True)
    assert len(receptacle_change) == len(objectId_list)
    return receptacle_change

def detect_type_state_change(instance_state_change, object_instance_list_sorted, object_type_list_sorted):
    '''detect object state change from one timestep to the next. By object type instead of object instance.'''
    state_change = []
    
    curr_typ = object_instance_list_sorted[0].split('|')[0]
    changes = []
    for i, instance_name in enumerate(object_instance_list_sorted):
        instance_typ = instance_name.split('|')[0]
        if instance_typ != curr_typ:
            state_change.append(True in changes)
            changes = []
            curr_typ = instance_typ
        changes.append(instance_state_change[i])
            
    # don't forget about last typ!
    state_change.append(int(True in changes))
    assert len(state_change) == len(object_type_list_sorted)
    return state_change

def detect_type_visibility(instance_visible, object_instance_list_sorted, object_type_list_sorted):
    '''detect if object type is visible'''
    type_visible = []
    
    curr_typ = object_instance_list_sorted[0].split('|')[0]
    visible = []
    for i, instance_name in enumerate(object_instance_list_sorted):
        instance_typ = instance_name.split('|')[0]
        if instance_typ != curr_typ:
            type_visible.append(True in visible)
            visible = []
            curr_typ = instance_typ
        visible.append(instance_visible[i])
            
    # don't forget about last typ!
    type_visible.append(int(True in visible))
    assert len(type_visible) == len(object_type_list_sorted)
    return type_visible

def get_receptacle_list(state, object_instance_list_sorted):
    '''make receptacleId list in the same order as object Id List'''
    # positions corresponds to end of task
    receptacleId_lookup = defaultdict(lambda: 'None')
    for o in state['objects_metadata']:
        if o['parentReceptacles']:
            receptacleId_lookup[o['objectId'].lower()] = o['parentReceptacles'][0].lower()
        else:
            receptacleId_lookup[o['objectId'].lower()] = 'none'

    receptacleId_list = []
    receptacleType_list = []    
    for i, instance_name in enumerate(object_instance_list_sorted):
        receptacleId = receptacleId_lookup[instance_name]
        receptacleId_list.append(receptacleId)
        receptacleType_list.append(receptacleId.split('|')[0])
    
    assert len(receptacleId_list) == len(object_instance_list_sorted)
    return receptacleId_list, receptacleType_list

def get_obj_distance(state, object_instance_list_sorted):
    '''get distance of object from agent.'''
    dist_lookup = defaultdict(lambda: 1e6)
    for o in state['objects_metadata']:
        dist_lookup[o['objectId'].lower()] = o['distance']
    dist_list = []
    for i, instance_name in enumerate(object_instance_list_sorted):
        dist_list.append(dist_lookup[instance_name])
    return dist_list


####################################################
## Check and Fix object state to subgoal alignment

def check_and_correct_alignments(extracted_features, traj_data):
    '''check and correct subgoal feature alignment.'''
    subgoal_len_features = [(i,len(subgoal)) for i, subgoal in enumerate(extracted_features['instance_visibile'])]
    counter_items = list(Counter([low_a['high_idx'] for low_a in traj_data['plan']['low_actions']]).items())

    if counter_items == subgoal_len_features[:-1]:
        corrected = False
    else:
        # last subgoal has one more timestep than expected
        assert subgoal_len_features[-1][1] == counter_items[-1][1] + 1, \
            'Alignment error.' # the last two subgoals were incorrectly lumped together during collection.

        # split the errorneous last subgoal back into two subgoal
        apply_alignment_correction(extracted_features, subgoal_count_specific_features)
        subgoal_len_features2 = [
            (i,len(subgoal)) for i, subgoal in enumerate(extracted_features['instance_visibile'])
        ]
        assert counter_items == subgoal_len_features2[:-1]
        corrected = True       
            
    return corrected

def apply_alignment_correction(extracted_feat, subgoal_count_specific_features):
    '''apply alignment correction to every subgoal'''
    for feat in subgoal_count_specific_features:
        extracted_feat[feat] = split_last_subgoal(extracted_feat[feat])
    
def split_last_subgoal(feature):
    '''
    split the errorneous last subgoal back into two subgoal
    feature: feature[subgoal_i][subgoal timestep] = [val1, val2, ...]
    '''
    aligned_feature = feature[:-1] + [feature[-1][:-1]] + [[feature[-1][-1]]]
    return aligned_feature

####################################################

def extract_states_for_model(shifted_states, saved_states, object_vocab):
    '''extract and numericalize object states for language models to auto-label'''
    
    # setup data structure
    objectId_list = sorted([o['objectId'].lower() for o in shifted_states[-1]['objects_metadata']])
    objectTyp_list = sorted(set([o.split('|')[0].lower() for o in objectId_list]))
    num_subgoals = len(set([s['subgoal_step'] for s in shifted_states]))

    feat = {
        'subgoal': [[] for _ in range(num_subgoals)],
        'objectInstanceList': objectId_list,
        'objectInstanceList_TypeNum': object_vocab['object_type'].word2index([o.split('|')[0] for o in objectId_list], train=True),
        # [subgoal_i][subgoal timestep] = [receptacle Id for ob in objectId_list]
        'objectReceptacleList': [[] for _ in range(num_subgoals)], 
        'objectReceptacleList_TypeNum': [[] for _ in range(num_subgoals)],
        # [subgoal_i][subgoal timestep] = [T/F for ob in objectId_list]
        'instance_visibile': [[] for _ in range(num_subgoals)], 
        'instance_state_change': [[] for _ in range(num_subgoals)],
        'instance_state_change_since_last_subgoal': [[] for _ in range(num_subgoals)],
        'instance_distance': [[] for _ in range(num_subgoals)],
        'instance_receptacle_change': [[] for _ in range(num_subgoals)],
        'instance_receptacle_change_since_last_subgoal': [[] for _ in range(num_subgoals)],
        'objectTypeList': objectTyp_list,
        'objectTypeList_TypeNum': object_vocab['object_type'].word2index(objectTyp_list, train=True),
        # [subgoal_i][subgoal timestep] = [T/F for ob in objectTyp_list]
        'type_visibile': [[] for _ in range(num_subgoals)], 
        'type_state_change': [[] for _ in range(num_subgoals)],
        'type_state_change_since_last_subgoal': [[] for _ in range(num_subgoals)],
    }

    last_obj_states = {obj['objectId'].lower():get_object_states(obj, object_state_types) for obj in saved_states[0]['objects_metadata']}
    last_subgoal_obj_states = None
    
    last_obj_receptacles = {obj['objectId'].lower():obj['parentReceptacles'] for obj in saved_states[0]['objects_metadata']}
    last_subgoal_obj_receptacles = None
    
    # go through the states one at a time to compute date structure
    for t, state in enumerate(shifted_states):

        subgoal_i = state['subgoal_step']
        subgoal = state['subgoal']

        if state['new_subgoal']:
            subgoal_t = 0
            last_subgoal_obj_states = last_obj_states
            last_subgoal_obj_receptacles = last_obj_receptacles

        feat['subgoal'][subgoal_i].append(subgoal)

        # get receptacles identity
        receptacle_list, receptacle_type_list = get_receptacle_list(state, objectId_list)
        receptacle_type_num = object_vocab['object_type'].word2index(receptacle_type_list, train=True)
        feat['objectReceptacleList'][subgoal_i].append(receptacle_list)
        feat['objectReceptacleList_TypeNum'][subgoal_i].append(receptacle_type_num)
        
        # get state change
        curr_obj_states = {obj['objectId'].lower():get_object_states(obj, object_state_types) for obj in state['objects_metadata']}
        
        # list same order as objectId_list
        state_change = detect_state_change(last_obj_states, curr_obj_states, objectId_list)
        state_change_since_last_subgoal = detect_state_change(last_subgoal_obj_states, curr_obj_states, objectId_list)
        last_obj_states = curr_obj_states
        
        feat['instance_state_change'][subgoal_i].append(state_change)
        feat['instance_state_change_since_last_subgoal'][subgoal_i].append(state_change_since_last_subgoal)
    
        # get instance distance
        obj_dist = get_obj_distance(state, objectId_list)
        
        feat['instance_distance'][subgoal_i].append(obj_dist)
        
        # get receptacles state
        curr_obj_receptacles = {obj['objectId'].lower():obj['parentReceptacles'] for obj in state['objects_metadata']}
        
        # get receptacle change
        receptacle_change = detect_receptacle_change(last_obj_receptacles, curr_obj_receptacles, objectId_list)
        receptacle_change_since_last_subgoal = detect_receptacle_change(last_subgoal_obj_receptacles, curr_obj_receptacles, objectId_list)
        last_obj_receptacles = curr_obj_receptacles
        
        feat['instance_receptacle_change'][subgoal_i].append(receptacle_change)
        feat['instance_receptacle_change_since_last_subgoal'][subgoal_i].append(receptacle_change_since_last_subgoal)
    
        # get visibility
        visible = [False for _ in objectId_list]
        for ob in state['objects_metadata']:
            pos = objectId_list.index(ob['objectId'].lower())
            visible[pos] = ob['visible']
        feat['instance_visibile'][subgoal_i].append(visible)
        
        # get type state change
        type_state_change = detect_type_state_change(state_change, objectId_list, objectTyp_list)
        feat['type_state_change'][subgoal_i].append(type_state_change)
        type_state_change_since_last_subgoal = detect_type_state_change(state_change_since_last_subgoal, objectId_list, objectTyp_list)
        feat['type_state_change_since_last_subgoal'][subgoal_i].append(type_state_change_since_last_subgoal)

        # get type visibility
        type_visible = detect_type_visibility(visible, objectId_list, objectTyp_list)
        feat['type_visibile'][subgoal_i].append(type_visible)

        subgoal_t += 1
        
    return feat

def main(parse_args):

    # load splits
    with open(parse_args.splits, 'r') as f:
        split = json.load(f)['augmentation']
    print('number of trajectories in split = ', len(split))

    # load object vocab
    with open(parse_args.object_vocab, 'r') as f:
        object_vocab = json.load(f)
    print('loaded object vocab:', object_vocab)

    # extract features and build data structure
    extract_feat_outpaths = []
    corrected = 0

    print(f'Start augmenting object states for {len(split)} trajectories.')

    for task in progressbar.progressbar(split):
        
        # load metadata
        metadata_outpath = os.path.join(parse_args.data, task['task'], 'metadata_states.json')
        with open(metadata_outpath, 'r') as f:
            saved_states = json.load(f)
        
        # load traj data
        pp_folder = 'pp_model:seq2seq_per_subgoal,name:v2_epoch_40_obj_instance_enc_max_pool_dec_aux_loss_weighted_bce_1to2'
        traj_data_p = os.path.join(parse_args.data, task['task'], pp_folder, f'demo_{task["repeat_idx"]}.json')
        with open(traj_data_p, 'r') as f:
            traj_data = json.load(f)

        # extract obj state features
        root = os.path.join(parse_args.data, task['task'])
        shifted_states = shift_states(saved_states)
        extract_feat = extract_states_for_model(shifted_states, saved_states, object_vocab)
        extract_feat['metadata_path'] = metadata_outpath
        extract_feat['root'] = root
        
        # check if extracted features align with traj data, apply correction
        corrected += int(check_and_correct_alignments(extract_feat, traj_data))
        
        # save out to same directory level
        extract_feat_outpath = metadata_outpath.replace('/metadata_states.json', '/extracted_feature_states.json')
        with open(extract_feat_outpath, 'w') as f:
            json.dump(extract_feat ,f)
        
        # copy of all output paths
        extract_feat_outpaths.append(extract_feat_outpath)

    print(f'Finished augmenting object states for {len(split)} trajectories.')

    # keep record of all the new json paths
    augmented_paths_dir = os.path.join(parse_args.data, 'pipeline_logs')
    if not os.path.exists(augmented_paths_dir):
        os.makedirs(augmented_paths_dir, exist_ok = True)
    augmented_paths_loc = os.path.join(augmented_paths_dir, 'preprocessed_object_state_paths.json')
    with open(augmented_paths_loc, 'wb') as f:
        json.dump(extract_feat_outpaths, f)
    print(f'Referecence: Saved paths of augmented object states to {augmented_paths_loc}')


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        '--data', help='dataset directory.', type=str
    )
    parser.add_argument(
        '--splits', help='json file containing trajectory splits.', type=str
    )
    parser.add_argument(
        '--object_vocab', type=str,
        help='object vocabulary.'
    )

    parse_args = parser.parse_args()

    main(parse_args)