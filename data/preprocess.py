import sys
import os
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'models'))

import re
import json
import revtok
import torch
import copy
import progressbar
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
        # if action_high:
        #     words = [w.strip().lower().replace('sink', 'sinkbasin').replace('bathtub', 'bathtubbasin') for w in words]
        # else:
        words = [w.strip().lower() for w in words]
        return vocab.word2index(words, train=train)


    def preprocess_splits(self, splits, preprocess_lang=True, train_vocab=True, save_vocab_to_dout=True, demo_mode=False):
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

            for task in progressbar.progressbar(d):
                
                # load json file
                json_path = os.path.join(self.args.data, '' if demo_mode else k, task['task'], 'traj_data.json')
                with open(json_path) as f:
                    ex = json.load(f)

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
                    self.process_language(ex, traj, r_idx, train=train_vocab, demo_mode=demo_mode)

                # numericalize actions for train/valid splits
                if 'test' not in k: # expert actions are not available for the test set
                    self.process_actions(ex, traj, train=train_vocab, language_processed=preprocess_lang)

                # check if preprocessing storage folder exists
                preprocessed_folder = os.path.join(self.args.data, task['task'], self.args.pp_folder)
                if not os.path.isdir(preprocessed_folder):
                    os.makedirs(preprocessed_folder)               

                # save preprocessed json
                if demo_mode:
                    preprocessed_json_path = os.path.join(preprocessed_folder, "demo_%d.json" % r_idx)
                else:
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

    def process_language(self, ex, traj, r_idx, train=True, demo_mode=False):
        # tokenize language
        ann_key = 'explainer_annotations' if demo_mode else 'turk_annotations'
        traj['ann'] = {
            'goal': revtok.tokenize(remove_spaces_and_lower(ex[ann_key]['anns'][r_idx]['task_desc'])) + ['<<goal>>'],
            'instr': [revtok.tokenize(remove_spaces_and_lower(x)) for x in ex[ann_key]['anns'][r_idx]['high_descs']] + [['<<stop>>']],
            'repeat_idx': r_idx
        }

        # numericalize language
        traj['num'] = {}
        traj['num']['lang_goal'] = self.numericalize(self.vocab['word'], traj['ann']['goal'], train=train)
        traj['num']['lang_instr'] = [self.numericalize(self.vocab['word'], x, train=train) for x in traj['ann']['instr']]


    def process_actions(self, ex, traj, train=True, language_processed=True):
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
        if 'num' not in traj.keys(): # if we skip process_language()
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
                'action_high_args': self.numericalize(self.vocab['action_high'], a['discrete_action']['args'], train=train, action_high=True),
            })

        # check alignment between step-by-step language and action sequence segments
        action_low_seg_len = len(traj['num']['action_low'])
        if language_processed:
            lang_instr_seg_len = len(traj['num']['lang_instr'])
            seg_len_diff = action_low_seg_len - lang_instr_seg_len
            if seg_len_diff != 0:
                assert (seg_len_diff == 1) # sometimes the alignment is off by one  ¯\_(ツ)_/¯
                self.merge_last_two_low_actions(traj)


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