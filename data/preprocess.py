import sys
import os
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'models'))

import re
import json
import revtok
import torch
import copy
import progressbar
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from vocab import Vocab
from model.seq2seq import Module as model
from gen.utils.py_util import remove_spaces_and_lower

import shutil

class Dataset(object):

    def __init__(self, args, vocab=None):
        self.args = args

        if vocab is None:
            self.vocab = {
                'word': Vocab(['<<pad>>', '<<seg>>', '<<goal>>']),
                'action_low': Vocab(['<<pad>>', '<<seg>>', '<<stop>>']),
                'action_high': Vocab(['<<pad>>', '<<seg>>', '<<stop>>']),
            }
        else:
            self.vocab = vocab

        self.word_seg = self.vocab['word'].word2index('<<seg>>', train=False)


    @staticmethod
    def numericalize(vocab, words, train=True, action_high=False):
        '''
        converts words to unique integers
        '''
        words = [w.strip().lower() for w in words]
        return vocab.word2index(words, train=train)


    def preprocess_splits(self, splits, preprocess_lang=True, train_vocab=True, save_vocab_to_dout=True, augmentation_mode=False):
        '''
        saves preprocessed data as jsons in specified folder
        splits : dict read from <data path>/splits/*.json files
        preprocess_lang : boolean. Whether to process language instructions.
        train_vocab : boolean. Whether to keep training vocab -- add new types to them.
        save_vocab_to_dout : boolean. Whether to save a copy of vocab to explainer model training directory. Use 'True' when training model. Use 'False' when preprocessing for demo.
        '''
        for k, d in splits.items():
            print('Preprocessing {}'.format(k))

            # debugging:
            if self.args.fast_epoch:
                d = d[:16]

            print(d)

            for task in progressbar.progressbar(d):
                
                # load json file
                json_path = os.path.join(self.args.data, '' if augmentation_mode else k, task['task'], 'traj_data.json')
                
                try:
                    with open(json_path) as f:
                        ex = json.load(f)
                except:
                    breakpoint()

                # copy trajectory
                # repeat_idx is the index of the annotation for each trajectory, none if generated from demo
                r_idx = task['repeat_idx']
                traj = ex.copy()

                # root & split
                traj['root'] = os.path.join(self.args.data, task['task'])
                traj['split'] = k
                traj['repeat_idx'] = r_idx

                # numericalize language
                if preprocess_lang:
                    self.process_language(ex, traj, r_idx, train=train_vocab)

                # numericalize actions for train/valid splits
                if 'test' not in k: # expert actions are not available for the test set
                    self.process_actions(ex, traj, train=train_vocab, language_already_processed=preprocess_lang)

                # check if preprocessing storage folder exists
                preprocessed_folder = os.path.join(self.args.data, task['task'], self.args.pp_folder)
                if not os.path.isdir(preprocessed_folder):
                    os.makedirs(preprocessed_folder)               
                
                # save preprocessed json
                preprocessed_json_path = os.path.join(preprocessed_folder, "ann_%d.json" % r_idx)
                with open(preprocessed_json_path, 'w') as f:
                    json.dump(traj, f, sort_keys=True, indent=4)

        # save vocab in dout path if explainer training
        if save_vocab_to_dout:
            if not os.path.exists(self.args.dout):
                os.mkdir(self.args.dout)
            vocab_dout_path = os.path.join(self.args.dout, '%s.vocab' % self.args.pp_folder)
            torch.save(self.vocab, vocab_dout_path)

        # save vocab in data path
        vocab_data_path = os.path.join(self.args.data, '%s.vocab' % self.args.pp_folder)
        torch.save(self.vocab, vocab_data_path)

    def preprocess_splits_augmentation(self, splits, lmtag, train_vocab=True):
        '''
        Preprocess newly sampled trajectories with explainer or baseline predicted instructions. 
        Augmented data is used for training only.
        lmtag: 'baseline', 'explainer_auxonly', 'explainer_enconly' or 'explainer_full'
        '''
        d = splits['augmentation']

        if self.args.fast_epoch:
            d = d[:16]
        
        for task in progressbar.progressbar(d):

            # make output preprocessed folder
            preprocessed_folder = os.path.join(self.args.data, task['task'], 'pp_' + lmtag)
            if not os.path.isdir(preprocessed_folder):
                os.makedirs(preprocessed_folder)

            # load raw traj file
            raw_traj_p = os.path.join(self.args.data, task['task'], 'traj_data.json')
            with open(raw_traj_p, 'r') as f:
                ex = json.load(f)
            traj = ex.copy()
            r_idx = 0

            # set root & split
            traj['root'] = os.path.join(self.args.data, task['task'])
            traj['split'] = 'train_aug'            

            # numericalize language
            self.process_language(ex, traj, r_idx, ann_key=lmtag+'_annotations')

            # numericalize actions
            self.process_actions(ex, traj, train=train_vocab, language_already_processed=True)            

            # save out
            preprocessed_json_path = os.path.join(preprocessed_folder, f"ann_{r_idx}.json")
            with open(preprocessed_json_path, 'w') as f:
                json.dump(traj, f, sort_keys=True, indent=4)

        # save vocab in data path
        vocab_data_path = os.path.join(self.args.data, '%s.vocab' % self.args.pp_folder)
        torch.save(self.vocab, vocab_data_path) 

    def process_language(self, ex, traj, r_idx, train=True, ann_key='turk_annotations'):
        '''tokenize language, save to traj['ann'], numeralize language tokens, save to traj['num']'''

        # tokenize and numerical language, save numericalized to traj
        tokenized_goal, tokenized_instrs = self.numeralize_instr(
            traj=traj,
            goal=ex[ann_key]['anns'][r_idx]['task_desc'], 
            instrs=ex[ann_key]['anns'][r_idx]['high_descs'],
            train=train)

        # save tokenize to traj
        traj['ann'] = {
            'goal': tokenized_goal,
            'instr': tokenized_instrs,
            'repeat_idx': r_idx
        }        

    def numeralize_instr(self, traj, goal, instrs, train=True):

        tokenized_goal = revtok.tokenize(remove_spaces_and_lower(goal)) + ['<<goal>>']
        tokenized_instrs = [revtok.tokenize(remove_spaces_and_lower(x)) for x in instrs] + [['<<stop>>']]

        # numericalize language
        if 'num' not in traj.keys():
            traj['num'] = {}
        traj['num']['lang_goal'] = self.numericalize(self.vocab['word'], tokenized_goal, train=train)
        traj['num']['lang_instr'] = [self.numericalize(self.vocab['word'], x, train=train) for x in tokenized_instrs]        

        return tokenized_goal, tokenized_instrs

    def process_actions(self, ex, traj, train=True, language_already_processed=True):
        # deal with missing end high-level action
        self.fix_missing_high_pddl_end_action(ex)

        # end action for low_actions
        end_action = {
            'api_action': {'action': 'NoOp'}, 
            'discrete_action': {'action': '<<stop>>', 'args': {}},
            'high_idx': ex['plan']['high_pddl'][-1]['high_idx']
        }

        # init action_low and action_high
        num_hl_actions = len(ex['plan']['high_pddl'])
        if 'num' not in traj.keys():
            traj['num'] = {}
        traj['num']['action_low'] = [list() for _ in range(num_hl_actions)]  # temporally aligned with HL actions
        traj['num']['action_high'] = []
        low_to_high_idx = []

        for a in (ex['plan']['low_actions'] + [end_action]):
            # high-level action index (subgoals)
            high_idx = a['high_idx']
            low_to_high_idx.append(high_idx)

            # low-level action (API commands)
            traj['num']['action_low'][high_idx].append({
                'high_idx': a['high_idx'],
                'action': self.vocab['action_low'].word2index(a['discrete_action']['action'], train=train),
                'action_high_args': a['discrete_action']['args'],
            })

            # low-level bounding box (not used in the model)
            if 'bbox' in a['discrete_action']['args']:
                xmin, ymin, xmax, ymax = [float(x) if x != 'NULL' else -1 for x in a['discrete_action']['args']['bbox']]
                traj['num']['action_low'][high_idx][-1]['centroid'] = [
                    (xmin + (xmax - xmin) / 2) / self.args.pframe,
                    (ymin + (ymax - ymin) / 2) / self.args.pframe,
                ]
            else:
                traj['num']['action_low'][high_idx][-1]['centroid'] = [-1, -1]

            # low-level interaction mask (Note: this mask needs to be decompressed)
            if 'mask' in a['discrete_action']['args']:
                mask = a['discrete_action']['args']['mask']
            else:
                mask = None
            traj['num']['action_low'][high_idx][-1]['mask'] = mask

            # interaction validity
            valid_interact = 1 if model.has_interaction(a['discrete_action']['action']) else 0
            traj['num']['action_low'][high_idx][-1]['valid_interact'] = valid_interact

        # low to high idx
        traj['num']['low_to_high_idx'] = low_to_high_idx

        # high-level actions
        for a in ex['plan']['high_pddl']:
            traj['num']['action_high'].append({
                'high_idx': a['high_idx'],
                'action': self.vocab['action_high'].word2index(a['discrete_action']['action'], train=train),
            })

        # fix if language and action segments are not aligned
        if language_already_processed:
            self.check_lang_action_segment_alignments(traj, apply_fix=True)

    def check_lang_action_segment_alignments(self, traj, apply_fix=True):
        '''
        check alignment between step-by-step language and action sequence segments
        '''
        action_low_seg_len = len(traj['num']['action_low'])
        lang_instr_seg_len = len(traj['num']['lang_instr'])
        seg_len_diff = action_low_seg_len - lang_instr_seg_len
        if seg_len_diff != 0:
            try:
                assert (seg_len_diff == 1) # sometimes the alignment is off by one  ¯\_(ツ)_/¯
            except:
                import pdb; pdb.set_trace()
            if apply_fix:
                self.merge_last_two_low_actions(traj)
            return False
        else:
            return True

    def fix_missing_high_pddl_end_action(self, ex):
        '''
        appends a terminal action to a sequence of high-level actions
        '''
        if ex['plan']['high_pddl'][-1]['planner_action']['action'] != 'End':
            ex['plan']['high_pddl'].append({
                'discrete_action': {'action': 'NoOp', 'args': []},
                'planner_action': {'value': 1, 'action': 'End'},
                'high_idx': len(ex['plan']['high_pddl'])
            })

    def merge_last_two_low_actions(self, conv):
        '''
        combines the last two action sequences into one sequence
        '''
        extra_seg = copy.deepcopy(conv['num']['action_low'][-2])
        for sub in extra_seg:
            sub['high_idx'] = conv['num']['action_low'][-3][0]['high_idx']
            conv['num']['action_low'][-3].append(sub)
        del conv['num']['action_low'][-2]
        conv['num']['action_low'][-1][0]['high_idx'] = len(conv['plan']['high_pddl']) - 1

def main(parse_args):
    '''Preprocess action tokens'''

    # load split
    with open(parse_args.splits, 'r') as f:
        splits = json.load(f)
    print(f'Loaded {len(splits["augmentation"])} number of trajectories in augmentation split.')

    if parse_args.action_tokens_only_for_instruction_labeling:

        # load model
        ckpt = torch.load(parse_args.model_path)
        print('Loaded language model and vocab')
        print('Vocab:', ckpt['vocab'])

        dataset = Dataset(parse_args, ckpt['vocab'])
        dataset.preprocess_splits(
            splits, preprocess_lang=False, train_vocab=False, save_vocab_to_dout=False, augmentation_mode=True)

        print('Finished preprocessing action tokens with language model vocab for auto-instruction labeling.')

    else:

        # load vocab
        if parse_args.optional_agent_vocab_path:
            vocab = torch.load(parse_args.optional_agent_vocab_path)
            train_vocab = True
        else:
            train_vocab = False
        print('Vocab:', vocab)

        dataset = Dataset(parse_args, vocab)
        dataset.preprocess_splits_augmentation(splits, parse_args.lmtag, train_vocab=train_vocab)       

        print('Finished preprocessing action and language tokens for agent training.')


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        '--data', help='dataset directory.', type=str
    )
    parser.add_argument(
        '--splits', help='json file containing trajectory splits.', type=str
    )
    parser.add_argument(
        '--fast_epoch', default=False, help='debug run with few examples.'
    )
    parser.add_argument(
        '--model_path', type=str, help='path to instruction generating language model .pth'
    )
    parser.add_argument(
        '--model_name', type=str, 
        help='model save name. \
            e.g. model_seq2seq_per_subgoal,name_v2_epoch_40_obj_instance_enc_max_pool_dec_aux_loss_weighted_bce_1to2'
    )
    parser.add_argument(
        '--action_tokens_only_for_instruction_labeling', action='store_true', 
        help='only preprocess action tokens using language model vocab for auto-instruction labeling.'
    )
    parser.add_argument(
        '--action_lang_tokens_for_agent_training', action='store_true', 
        help='preprocess both language and action tokens using agent vocab for agent training.'
    )
    parser.add_argument(
        '--optional_agent_vocab_path', default=None, 
        help='optional agent vocab referece. default is None. e.g. pp.vocab'
    )
    parser.add_argument(
        '--lmtag', type=str,
        help='language model tag. "baseline", "explainer_auxonly", "explainer_enconly" or "explainer_full"'
    )    

    parse_args = parser.parse_args()
    parse_args.pframe = 300
    parse_args.pp_folder = 'pp_' + parse_args.model_name

    assert parse_args.action_tokens_only_for_instruction_labeling or parse_args.action_lang_tokens_for_agent_training
    assert not (
        parse_args.action_tokens_only_for_instruction_labeling and parse_args.action_lang_tokens_for_agent_training
    )

    main(parse_args)