import os
import sys
sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'models'))

import torch
import pprint
import json
from data.preprocess import Dataset
from importlib import import_module
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from models.utils.helper_utils import optimizer_to


if __name__ == '__main__':
    # parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    # settings
    parser.add_argument('--seed', help='random seed', default=123, type=int)
    parser.add_argument('--data', help='dataset folder', default='data/json_feat_2.1.0')
    parser.add_argument('--splits', help='json file containing train/dev/test splits', default='data/splits/may17.json')
    parser.add_argument('--preprocess', help='store preprocessed data to json files', action='store_true')
    parser.add_argument('--pp_folder', help='folder name for preprocessed data', default='pp')
    parser.add_argument('--object_vocab', help='object_vocab version, should be file with .object_vocab ending. default is none', default='none')
    parser.add_argument('--save_every_epoch', help='save model after every epoch (warning: consumes a lot of space)', action='store_true')
    parser.add_argument('--model', help='model to use', default='seq2seq_nl_baseline')
    parser.add_argument('--gpu', help='use gpu', action='store_true')
    parser.add_argument('--dout', help='where to save model', default='exp/model:{model}')
    parser.add_argument('--resume', help='load a checkpoint')

    # hyper parameters
    parser.add_argument('--batch', help='batch size', default=8, type=int)
    parser.add_argument('--epoch', help='number of epochs', default=20, type=int)
    parser.add_argument('--lr', help='optimizer learning rate', default=1e-4, type=float)
    parser.add_argument('--decay_epoch', help='num epoch to adjust learning rate', default=10, type=int)
    parser.add_argument('--dhid', help='hidden layer size', default=512, type=int)
    parser.add_argument('--dframe', help='image feature vec size', default=2500, type=int)
    parser.add_argument('--demb', help='language embedding size', default=100, type=int)
    parser.add_argument('--pframe', help='image pixel size (assuming square shape eg: 300x300)', default=300, type=int)
    parser.add_argument('--mask_loss_wt', help='weight of mask loss', default=1., type=float)
    parser.add_argument('--action_loss_wt', help='weight of action loss', default=1., type=float)
    parser.add_argument('--subgoal_aux_loss_wt', help='weight of subgoal completion predictor', default=0., type=float)
    parser.add_argument('--pm_aux_loss_wt', help='weight of progress monitor', default=0., type=float)

    # architecture ablations
    parser.add_argument('--encoder_addons', type=str, default='none', choices=['none', 'max_pool_obj', 'biattn_obj'])
    parser.add_argument('--decoder_addons', type=str, default='none', choices=['none', 'aux_loss'])
    parser.add_argument('--object_repr', type=str, default='type', choices=['none', 'type', 'instance'])
    parser.add_argument('--reweight_aux_bce', help='reweight binary CE for auxiliary tasks', action='store_true')

    # target
    parser.add_argument('--predict_high_level_goal', help='predict abstract single high level goal for entire task.', action='store_true')

    # dropouts
    parser.add_argument('--zero_goal', help='zero out goal language', action='store_true')
    parser.add_argument('--zero_instr', help='zero out step-by-step instr language', action='store_true')
    parser.add_argument('--act_dropout', help='dropout rate for action input sequence', default=0., type=float)
    parser.add_argument('--lang_dropout', help='dropout rate for language (goal + instr)', default=0., type=float)
    parser.add_argument('--input_dropout', help='dropout rate for concatted input feats', default=0., type=float)
    parser.add_argument('--vis_dropout', help='dropout rate for Resnet feats', default=0.3, type=float)
    parser.add_argument('--hstate_dropout', help='dropout rate for LSTM hidden states during unrolling', default=0.3, type=float)
    parser.add_argument('--attn_dropout', help='dropout rate for attention', default=0., type=float)
    parser.add_argument('--actor_dropout', help='dropout rate for actor fc', default=0., type=float)
    parser.add_argument('--word_dropout', help='dropout rate for word fc', default=0., type=float)

    # other settings
    parser.add_argument('--train_teacher_forcing', help='use gpu', action='store_true')
    parser.add_argument('--train_student_forcing_prob', help='bernoulli probability', default=0.1, type=float)
    parser.add_argument('--temp_no_history', help='use gpu', action='store_true')

    # debugging
    parser.add_argument('--fast_epoch', help='fast epoch during debugging', action='store_true')
    parser.add_argument('--dataset_fraction', help='use fraction of the dataset for debugging (0 indicates full size)', default=0, type=int)
    
    # args and init
    args = parser.parse_args()
    args.dout = args.dout.format(**vars(args))
    torch.manual_seed(args.seed)

    # check if dataset has been preprocessed
    if not os.path.exists(os.path.join(args.data, "%s.vocab" % args.pp_folder)) and not args.preprocess:
        raise Exception("Dataset not processed; run with --preprocess")

    # make output dir
    pprint.pprint(args)
    if not os.path.isdir(args.dout):
        os.makedirs(args.dout)

    # load train/valid/tests splits
    with open(args.splits) as f:
        splits = json.load(f)
        pprint.pprint({k: len(v) for k, v in splits.items()})

    # preprocess and save
    if args.preprocess:
        print("\nPreprocessing dataset and saving to %s folders ... This will take a while. Do this once as required." % args.pp_folder)
        dataset = Dataset(args, None)
        dataset.preprocess_splits(splits)
        vocab = torch.load(os.path.join(args.dout, "%s.vocab" % args.pp_folder))
    else:
        print("skipping preprocessing.")
        vocab = torch.load(os.path.join(args.data, "%s.vocab" % args.pp_folder))

    # load object vocab
    if args.object_vocab != 'none':
        object_vocab = torch.load(os.path.join(args.data, '%s' % args.object_vocab)) 
    else:
        object_vocab = None

    # load model
    M = import_module('model.{}'.format(args.model))
    if args.resume:
        print("Loading: " + args.resume)
        model, optimizer, start_epoch, start_iters = M.Module.load(args.resume)
        end_epoch = args.epoch
        if start_epoch >= end_epoch:
            print('Checkpoint already finished {}/{} epochs.'.format(start_epoch, end_epoch))
            sys.exit(0)
        else:
            print("Restarting at epoch {}/{}".format(start_epoch, end_epoch-1))
    else:
        model = M.Module(args, vocab, object_vocab)
        optimizer = None
        start_epoch = 0
        start_iters = None
        end_epoch = args.epoch

    # to gpu
    if args.gpu:
        model = model.to(torch.device('cuda'))
        model.demo_mode = False
        if not optimizer is None:
            optimizer_to(optimizer, torch.device('cuda'))

    # start train loop
    model.run_train(splits, optimizer=optimizer, start_epoch=start_epoch, end_epoch=end_epoch, start_iters=start_iters)