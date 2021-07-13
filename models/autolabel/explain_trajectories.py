import os
import sys
sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'models'))

import re
import time
import torch
import pprint
import json
from collections import defaultdict

from data.preprocess import Dataset
from importlib import import_module
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser


def make_overrride_args(args, level='subgoal'):
    '''make the news args that override the old args loaded in model checkpoints'''
    assert level in ['subgoal', 'goal']
    new_args = {arg:getattr(args, arg) for arg in vars(args)}
    new_args['pp_folder'] = 'pp_{}'.format(
        re.findall('(model:.*,name.*)/', new_args[f'{level}_level_explainer_checkpt_path'])[0]
    )
    return new_args

def load_task_json(model, task):
    '''
    load preprocessed demo json from disk
    '''    
    json_path = os.path.join(model.args.data, task['task'], 'traj_data.json')
    retry = 0
    while True:
        try:
            if retry > 0:
                print ('load task retrying {}'.format(retry))
            with open(json_path, 'r') as f:
                data = json.load(f)
            return data
        except:
            retry += 1
            time.sleep(5)
            pass

def overwrite_task_json(model, task, ex):
    '''
    overwrite traj json data with predicted language included.
    '''
    json_path = os.path.join(model.args.data, task['task'], 'traj_data.json')
    retry = 0
    while True:
        try:
            if retry > 0:
                print ('load task for rewrite retrying {}'.format(retry))
            with open(json_path, 'w') as f:
                json.dump(ex, f)
            return
        except:
            retry += 1
            time.sleep(5)
            pass
      
@torch.no_grad()
def run_pred(split, model, batch_size):
    '''make prediction on data without loss or metrics computations'''
    p_split = {}
    model.eval()
    for batch, feat, _ in model.iterate(split, batch_size):
        out = model.forward(feat, validate_teacher_forcing=False, validate_sample_output=False)
        preds = model.extract_preds(out, batch, feat)
        p_split.update(preds)
    return p_split   

def make_demo_output(preds, data, model, level='subgoal'):
    '''
    make language output for demo
    '''
    outputs = {}
    for task in data:
        ex = load_task_json(model, task)
        # just the task_id 'traj_T...'
        i = model.get_task_and_ann_id(ex)
        outputs[i] = {'p_lang_instr': preds[i]['lang_instr'], 'task':task['task']}
        print(f'\n\n\n\nTASK {i} {level} level: {preds[i]["lang_instr"]}\n\n')
    return outputs

def write_ann_to_traj(preds, data, model, lmtag, level='subgoal'):
    '''write predicted language back to trajectory on disk'''

    assert level in ['subgoal', 'goal']
    ann_key = lmtag + '_annotations'

    for task in data:
        ex = load_task_json(model, task)
        i = model.get_task_and_ann_id(ex)
        if ann_key not in ex.keys():
            ex[ann_key] = {'anns':[{'task_desc':'', 'high_descs':[]}]}

        if level == 'subgoal':
            num_subgoal = max(preds[i]['lang_instr'].keys()) + 1
            ex[ann_key]['anns'][0]['high_descs'] = [preds[i]['lang_instr'][subgoal_j] for subgoal_j in range(num_subgoal)]
        else:
            ex[ann_key]['anns'][0]['task_desc'] = preds[i]['lang_instr']
        
        overwrite_task_json(model, task, ex)

def pred_and_save(split, split_name, model, batch_size, dout, lmtag, level='subgoal', debug=False):

    assert level in ['subgoal', 'goal']
    print(f'Processing {split_name} split with {level} level explainer model.')

    # model make predictions
    split_preds = run_pred(split, model, batch_size)
    # make language output for demo
    split_outputs = make_demo_output(split_preds, split, model, level)
    # save outputs to path for inspection
    pred_out_path = os.path.join(dout, f'{split_name}_{level}_output_preds.json')
    with open(pred_out_path, 'w') as f:
        json.dump(split_outputs, f)
    print(f'Saved {level} level language instruction outputs for demo to {pred_out_path}')

    # write predicted language back to trajectory on disk
    write_ann_to_traj(split_preds, split, model, lmtag, level)
    print(f'Overwrote traj data on disk with {level} level language instruction included.')

    # make detailed file to debug predictions for 
    # input, lang, attn, aux targets, ...
    if debug:
        split_debugs = model.make_debug(split_preds, split)
        pred_debug_path = os.path.join(dout, f'{split_name}_{level}_debug_preds.json')
        with open(pred_debug_path, 'w') as f:
            json.dump(split_debugs, f)
        print(f'Saving {level} level debug outputs for demo to {pred_debug_path}')

@torch.no_grad()
def main(args, splits, subgoal_level_model, goal_level_model=None):
    
    if args.fast_epoch:
        splits = {split_name:splits[split_name][:16] for split_name in splits.keys()}

    # Run explainer for splits
    for split_name in splits.keys():
        split = splits[split_name]

        if goal_level_model is not None:
            pred_and_save(
                split, split_name, goal_level_model, args.batch, args.dout, args.lmtag, level='goal', debug=args.debug
            )
        pred_and_save(
            split, split_name, subgoal_level_model, args.batch, args.dout, args.lmtag, level='subgoal', debug=args.debug
        )


if __name__ == '__main__':

    # parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    # settings
    parser.add_argument('--seed', help='random seed', default=123, type=int)
    parser.add_argument('--data', help='dataset folder')
    parser.add_argument('--splits', help='json file containing train/dev/test splits')
    parser.add_argument('--subgoal_level_explainer_checkpt_path', help='path to model checkpoint')
    parser.add_argument('--goal_level_explainer_checkpt_path', help='path to model checkpoint', default='None')
    parser.add_argument('--gpu', help='use gpu', action='store_true')
    parser.add_argument('--batch', help='batch size', default=8, type=int)

    # explainer or baseline
    parser.add_argument('--lmtag', help='language model name. e.g. baseline, explainer, explainer_auxonly, explainer_enconly', type=str)

    # debugging
    parser.add_argument('--fast_epoch', help='fast epoch during debugging', action='store_true')
    parser.add_argument('--debug', dest='debug', action='store_true')

    # args and init
    args = parser.parse_args()
    args.predict_goal_level_instruction = False
    torch.manual_seed(args.seed)

    print('Input args:')
    pprint.pprint(args)

    # load train/valid/tests splits
    with open(args.splits) as f:
        splits = json.load(f)
        pprint.pprint({k: len(v) for k, v in splits.items()})    

    # load model types
    print(f'Subgoal-level Explainer Model path : {args.subgoal_level_explainer_checkpt_path}')
    print(f'Goal-level Explainer Model path : {args.goal_level_explainer_checkpt_path}')

    # make output dir for predictions
    subgoal_mod_name = re.findall('(model:.*,name.*)/', args.subgoal_level_explainer_checkpt_path)[0]
    goal_mod_name = re.findall('(model:.*,name.*)/', args.goal_level_explainer_checkpt_path)[0] if args.goal_level_explainer_checkpt_path != 'None' else 'None'
    args.dout = os.path.join(args.data, f'dout_explainer_subgoal_{subgoal_mod_name}_goal_{goal_mod_name}')
    if not os.path.isdir(args.dout):
        print (f'Output directory: {args.dout}')
        os.makedirs(args.dout)

    # load models modules
    subgoal_module = re.findall('model:(.*),name:', args.subgoal_level_explainer_checkpt_path)[0]
    goal_module = re.findall('model:(.*),name:', args.goal_level_explainer_checkpt_path)[0] if args.goal_level_explainer_checkpt_path != 'None' else 'None'

    # load subgoal-level model
    print(f'Loading subgoal-level Explainer Model module : {subgoal_module}')
    M_subgoal = import_module('model.{}'.format(subgoal_module))
    # load model checkpoint, override path related arguments
    subgoal_level_model, _, _, _ = M_subgoal.Module.load(
        args.subgoal_level_explainer_checkpt_path, make_overrride_args(args, 'subgoal'))
    subgoal_level_model.demo_mode = True

    # load goal-level model
    goal_level_model = None
    if args.goal_level_explainer_checkpt_path != 'None':
        print(f'Loading goal-level Explainer Model module : {goal_module}')
        args.predict_goal_level_instruction = True
        M_goal = import_module('model.{}'.format(goal_module))
        # load model checkpoint, override path related arguments
        goal_level_model, _, _, _ = M_goal.Module.load(
            args.goal_level_explainer_checkpt_path, make_overrride_args(args, 'goal'))
        goal_level_model.demo_mode = True

    if args.gpu:
        subgoal_level_model = subgoal_level_model.to(torch.device('cuda'))
        if args.goal_level_explainer_checkpt_path != 'None':
            goal_level_model = goal_level_model.to(torch.device('cuda'))

    main(args, splits, subgoal_level_model, goal_level_model)
