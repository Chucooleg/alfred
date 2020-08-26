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
from tensorboardX import SummaryWriter


def make_overrride_args(args, level='low'):
    '''make the news args that override the old args loaded in model checkpoints'''
    assert level in ['low', 'high']
    new_args = {arg:getattr(args, arg) for arg in vars(args)}
    # use new pp folder: pp_model:.....,name:.....
    new_args['pp_folder'] = 'pp_{}'.format(re.findall('(model:.*,name.*)/', new_args[f'{level}_level_explainer_checkpt_path'])[0])
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
                import pdb; pdb.set_trace()
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
      

def validate_vocab(args, model, level='low'):
    # validate language vocab
    if level == 'high' and args.high_level_explainer_checkpt_path == 'None':
        return
    else:
        # load vocab from data
        pp_suffix = re.findall('(model:.*,name.*)/', args.low_level_explainer_checkpt_path if level=='low' else args.high_level_explainer_checkpt_path)[0]
        data_vocab_path = os.path.join(args.data, f'pp_{pp_suffix}.vocab')
        assert os.path.exists(data_vocab_path), 'data vocab path does not exist, cannot pass sanity check'
        assert torch.load(data_vocab_path) == model.vocab, 'data vocab and model vocab do not match, cannot pass sanity check'

    if level == 'low' and not args.baseline:
        # validate object vocab
        data_obj_vocab_path = os.path.join(args.data, f'pp_{pp_suffix}.object_vocab')
        if (model.object_vocab is not None) or (os.path.exists(data_obj_vocab_path)):
            assert torch.load(data_obj_vocab_path)['object_type'].__dict__['_index2word'] == model.object_vocab['object_type'].__dict__['_index2word'], 'data vocab object and model object vocab do not match, cannot pass sanity check'

@torch.no_grad()
def run_demo_pred(split, model, batch_size):
    '''make prediction on data without loss or metrics computations'''
    p_split = {}
    model.eval()
    for batch, feat, _ in model.iterate(split, batch_size):
        out = model.forward(feat, validate_teacher_forcing=False, validate_sample_output=False)
        preds = model.extract_preds(out, batch, feat)
        p_split.update(preds)
    return p_split   

def make_demo_output(preds, data, model, level='low'):
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


def write_ann_to_traj(preds, data, model, baseline=False, level='low'):
    '''write predicted language back to trajectory on disk'''

    assert level in ['low', 'high']
    ann_key = 'baseline_annotations' if baseline else 'explainer_annotations'

    for task in data:
        ex = load_task_json(model, task)
        i = model.get_task_and_ann_id(ex)
        if ann_key not in ex.keys():
            ex[ann_key] = {'anns':[{'task_desc':'', 'high_descs':[]}]}

        if level == 'low':
            num_subgoal = max(preds[i]['lang_instr'].keys()) + 1
            ex[ann_key]['anns'][0]['high_descs'] = [preds[i]['lang_instr'][subgoal_j] for subgoal_j in range(num_subgoal)]
        else:
            ex[ann_key]['anns'][0]['task_desc'] = preds[i]['lang_instr']
        
        overwrite_task_json(model, task, ex)


def pred_and_save(split, split_name, model, batch_size, dout, baseline=False, level='low', debug=False):
    assert level in ['low', 'high']

    print(f'Processing {split_name} split with {level} level explainer model.')

    # model make predictions
    split_preds = run_demo_pred(split, model, batch_size)
    # make language output for demo
    split_outputs = make_demo_output(split_preds, split, model, level)
    # save outputs to path for next step in pipeline
    # /data_alfred/demo_generated/new_trajectories/dout_explainer_low_{}_high_{}/{split}_{level}_output_preds.json
    pred_out_path = os.path.join(dout, f'{split_name}_{level}_output_preds.json')
    with open(pred_out_path, 'w') as f:
        json.dump(split_outputs, f)
    print(f'Saving {level} level language instruction outputs for demo to {pred_out_path}')

    # write predicted language back to trajectory on disk
    write_ann_to_traj(split_preds, split, model, baseline, level)
    print(f'Overwrote traj data on disk with {level} level language instruction included.')

    # make file to debug predictions for 
    # input, lang, attn, aux targets, ...
    if debug:
        split_debugs = model.make_debug(split_preds, split)
        pred_debug_path = os.path.join(dout, f'{split_name}_{level}_debug_preds.json')
        with open(pred_debug_path, 'w') as f:
            json.dump(split_debugs, f)
        print(f'Saving {level} level debug outputs for demo to {pred_debug_path}')

@torch.no_grad()
def main(args, splits, low_level_model, high_level_model=None):
    
    if args.fast_epoch:
        splits = {split_name:splits[split_name][:16] for split_name in splits.keys()}

    # Run low level explainer for splits
    for split_name in splits.keys():
        split = splits[split_name]
        if high_level_model is not None:
            pred_and_save(split, split_name, high_level_model, args.batch, args.dout, baseline=args.baseline, level='high', debug=args.debug)
        pred_and_save(split, split_name, low_level_model, args.batch, args.dout, baseline=args.baseline, level='low', debug=args.debug)


if __name__ == '__main__':
    # parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    # settings
    parser.add_argument('--seed', help='random seed', default=123, type=int)
    parser.add_argument('--data', help='dataset folder', default='/root/data_alfred/demo_generated/new_trajectories')
    parser.add_argument('--splits', help='json file containing train/dev/test splits', default='/root/data_alfred/splits/demo_june13.json')
    parser.add_argument('--low_level_explainer_checkpt_path', help='path to model checkpoint')
    parser.add_argument('--high_level_explainer_checkpt_path', help='path to model checkpoint', default='None')
    parser.add_argument('--gpu', help='use gpu', action='store_true')
    parser.add_argument('--batch', help='batch size', default=8, type=int)

    # explainer or baseline
    parser.add_argument('--baseline', help='predicting with baseline (true) or explainer (false)', action='store_true')

    # debugging
    parser.add_argument('--fast_epoch', help='fast epoch during debugging', action='store_true')
    parser.add_argument('--debug', dest='debug', action='store_true')

    # args and init
    args = parser.parse_args()
    args.predict_high_level_goal = False
    torch.manual_seed(args.seed)

    print('Input args:')
    pprint.pprint(args)

    # load train/valid/tests splits
    with open(args.splits) as f:
        splits = json.load(f)
        pprint.pprint({k: len(v) for k, v in splits.items()})    

    # load model types
    print(f'Low-level Explainer Model path : {args.low_level_explainer_checkpt_path}')
    print(f'High-level Explainer Model path : {args.high_level_explainer_checkpt_path}')

    # make output dir for predictions
    # e.g. model:seq2seq_per_subgoal,name:v2_epoch_40_obj_instance_enc_max_pool_dec_aux_loss_weighted_bce_1to2
    low_mod_name = re.findall('(model:.*,name.*)/', args.low_level_explainer_checkpt_path)[0]
    high_mod_name = re.findall('(model:.*,name.*)/', args.high_level_explainer_checkpt_path)[0] if args.high_level_explainer_checkpt_path != 'None' else 'None'
    args.dout = os.path.join(args.data, f'dout_explainer_low_{low_mod_name}_high_{high_mod_name}')
    if not os.path.isdir(args.dout):
        print (f'Output directory: {args.dout}')
        os.makedirs(args.dout)

    # load models modules
    # e.g. seq2seq_per_subgoal
    low_module = re.findall('model:(.*),name:', args.low_level_explainer_checkpt_path)[0]
    high_module = re.findall('model:(.*),name:', args.high_level_explainer_checkpt_path)[0] if args.high_level_explainer_checkpt_path != 'None' else 'None'

    # load low-level model
    print(f'Loading low-level Explainer Model module : {low_module}')
    M_low = import_module('model.{}'.format(low_module))
    # load model checkpoint, override path related arguments
    low_level_model, _, _, _ = M_low.Module.load(
        args.low_level_explainer_checkpt_path, make_overrride_args(args, 'low'))
    low_level_model.demo_mode = True

    # load high-level model
    high_level_model = None
    if args.high_level_explainer_checkpt_path != 'None':
        print(f'Loading high-level Explainer Model module : {high_module}')
        args.predict_high_level_goal = True
        M_high = import_module('model.{}'.format(high_module))
        # load model checkpoint, override path related arguments
        high_level_model, _, _, _ = M_high.Module.load(
            args.high_level_explainer_checkpt_path, make_overrride_args(args, 'high'))
        high_level_model.demo_mode = True

    # make sure model vocab is the same vocab used to preprocess data
    validate_vocab(args, low_level_model, level='low')
    validate_vocab(args, high_level_model, level='high')

    # to gpu
    if args.gpu:
        low_level_model = low_level_model.to(torch.device('cuda'))
        if args.high_level_explainer_checkpt_path != 'None':
            high_level_model = high_level_model.to(torch.device('cuda'))

    # run evaluation
    main(args, splits, low_level_model, high_level_model)

