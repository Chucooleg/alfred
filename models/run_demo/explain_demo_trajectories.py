import os
import sys
sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'models'))

import re
import torch
import pprint
import json
from data.preprocess import Dataset
from importlib import import_module
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from tensorboardX import SummaryWriter


def run_demo_pred(split, model, args):
    '''make prediction on data without loss or metrics computations'''
    p_split = {}
    for batch, feat in self.iterate(split, args.batch):
        out = model.forward(feat, validate_teacher_forcing=False, validate_sample_output=False)
        preds = self.extract_preds(out, batch, feat)
        p_split.update(preds)
    return p_split   

def make_demo_output(preds, data):
    '''
    make language output for demo
    '''
    outputs = {}
    for task in data:
        ex = model.load_task_json(task)
        i = model.get_task_and_ann_id(ex)
        outputs[i] = {'p_lang_instr': preds[i]['lang_instr']}
    return outputs

@torch.no_grad()
def main(splits, model, args=None):

    args = args or model.args
    
    if args.fast_epoch:
        splits = {split_name:splits[split_name][:16] for split_name in splits.keys()}

    # display dout
    print("Saving model predictions to: %s" % args.dout)

    for split in splits:
        # model make predictions
        split_preds = run_demo_pred(split, model, args)
        # make language output for demo
        split_outputs = make_demo_output(split_preds, split)
        # save outputs to path for next step in pipeline
        pred_out_path = os.path.join(args.dout, f'{split}_output_preds.json')
        with open(pred_out_path, 'w') as f:
            json.dump(split_outputs, f)
        print(f'Saving language output for demo to {pred_out_path}')

        # make file to debug predictions for 
        # input, lang, attn, aux targets, ...
        if args.debug:
            split_debugs = model.make_debug(split_preds, split)
            pred_debug_path = os.path.join(args.dout, f'{split}_debug_preds.json')
            with open(pred_debug_path, 'w') as f:
                json.dump(split_debugs, f)
            print(f'Saving debugging output for demo to {pred_debug_path}')


if __name__ == '__main__':
    # parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    # settings
    # parser.add_argument('--seed', help='random seed', default=123, type=int)
    parser.add_argument('--data', help='dataset folder', default='/root/data_alfred/demo_generated/new_trajectories')
    parser.add_argument('--splits', help='json file containing train/dev/test splits', default='/root/data_alfred/splits/demo_june13.json')
    # parser.add_argument('--preprocess', help='store preprocessed data to json files', action='store_true')
    parser.add_argument('--pp_folder', help='folder name for preprocessed data, such as "pp_<modelname>" ')
    # parser.add_argument('--object_vocab', help='object_vocab version', default='object_20200521')
    # parser.add_argument('--save_every_epoch', help='save model after every epoch (warning: consumes a lot of space)', action='store_true')
    # parser.add_argument('--monitor_train_every', help='save debugging json and compute metric for training set at every regular interval. (warning: adds a lot of time)', default=100, type=int)
    parser.add_argument('--model_path', help='path to model checkpoint')
    parser.add_argument('--gpu', help='use gpu', action='store_true')
    parser.add_argument('--dout', help='where to save model predictions', default='exp/model:{model}')
    # parser.add_argument('--resume', help='load a checkpoint') 

    # hyper parameters
    # parser.add_argument('--batch', help='batch size', default=8, type=int)
    # parser.add_argument('--epoch', help='number of epochs', default=20, type=int)
    # parser.add_argument('--lr', help='optimizer learning rate', default=1e-4, type=float)
    # parser.add_argument('--decay_epoch', help='num epoch to adjust learning rate', default=10, type=int)
    # parser.add_argument('--dhid', help='hidden layer size', default=512, type=int)
    # parser.add_argument('--dframe', help='image feature vec size', default=2500, type=int)
    # parser.add_argument('--demb', help='language embedding size', default=100, type=int)
    # parser.add_argument('--pframe', help='image pixel size (assuming square shape eg: 300x300)', default=300, type=int)
    # parser.add_argument('--mask_loss_wt', help='weight of mask loss', default=1., type=float)
    # parser.add_argument('--action_loss_wt', help='weight of action loss', default=1., type=float)
    # parser.add_argument('--subgoal_aux_loss_wt', help='weight of subgoal completion predictor', default=0., type=float)
    # parser.add_argument('--pm_aux_loss_wt', help='weight of progress monitor', default=0., type=float)

    # # architecture ablations
    # parser.add_argument('--encoder_addons', type=str, default='none', choices=['none', 'max_pool_obj', 'biattn_obj'])
    # parser.add_argument('--decoder_addons', type=str, default='none', choices=['none', 'aux_loss'])
    # parser.add_argument('--object_repr', type=str, default='type', choices=['type', 'instance'])
    # parser.add_argument('--reweight_aux_bce', help='reweight binary CE for auxiliary tasks', action='store_true')

    # # dropouts
    # parser.add_argument('--zero_goal', help='zero out goal language', action='store_true')
    # parser.add_argument('--zero_instr', help='zero out step-by-step instr language', action='store_true')
    # parser.add_argument('--act_dropout', help='dropout rate for action input sequence', default=0., type=float)
    # parser.add_argument('--lang_dropout', help='dropout rate for language (goal + instr)', default=0., type=float)
    # parser.add_argument('--input_dropout', help='dropout rate for concatted input feats', default=0., type=float)
    # parser.add_argument('--vis_dropout', help='dropout rate for Resnet feats', default=0.3, type=float)
    # parser.add_argument('--hstate_dropout', help='dropout rate for LSTM hidden states during unrolling', default=0.3, type=float)
    # parser.add_argument('--attn_dropout', help='dropout rate for attention', default=0., type=float)
    # parser.add_argument('--actor_dropout', help='dropout rate for actor fc', default=0., type=float)
    # parser.add_argument('--word_dropout', help='dropout rate for word fc', default=0., type=float)

    # # other settings
    # parser.add_argument('--train_teacher_forcing', help='use gpu', action='store_true')
    # parser.add_argument('--train_student_forcing_prob', help='bernoulli probability', default=0.1, type=float)
    # parser.add_argument('--temp_no_history', help='use gpu', action='store_true')

    # debugging
    parser.add_argument('--fast_epoch', help='fast epoch during debugging', action='store_true')
    parser.add_argument('--debug', dest='debug', action='store_true')
    # parser.add_argument('--dataset_fraction', help='use fraction of the dataset for debugging (0 indicates full size)', default=0, type=int)

    # args and init
    args = parser.parse_args()
    args.dout = args.dout.format(**vars(args))
    torch.manual_seed(args.seed)

    # make output dir
    print('Input args:')
    pprint.pprint(args)
    if not os.path.isdir(args.dout):
        os.makedirs(args.dout)

    # load train/valid/tests splits
    with open(args.splits) as f:
        splits = json.load(f)
        pprint.pprint({k: len(v) for k, v in splits.items()})    

    # load model module from model path
    print(f'Model path : {args.model_path}')
    args.model = re.findall('model:(.*),name:', args.model_path)[0]
    print(f'Model module : {args.model}')
    M = import_module('model.{}'.format(args.model))

    # load model checkpoint, override path related arguments
    model, _, _, _ = M.Module.load(
        args.model_path, {arg:getattr(args, arg) for arg in vars(args)})

    # to gpu
    if args.gpu:
        model = model.to(torch.device('cuda'))

    # run evaluation
    main(splits, model, args)