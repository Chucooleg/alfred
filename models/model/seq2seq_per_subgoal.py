import os
import torch
import numpy as np
import nn.vnn as vnn
import collections
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from model.seq2seq import Module as Base
from models.utils.metric import compute_f1, compute_exact
from nltk.translate.bleu_score import sentence_bleu

# time
import time
from collections import defaultdict

class Module(Base):

    def __init__(self, args, vocab):
        '''
        Seq2Seq agent
        '''
        super().__init__(args, vocab)

        # encoder and self-attention
        encoder = vnn.ActionFrameAttnEncoderPerSubgoal
        self.enc = encoder( self.emb_action_low, args.dframe, args.dhid,
                            act_dropout=args.act_dropout,
                            vis_dropout=args.vis_dropout,
                            bidirectional=True)

        # language decoder
        decoder = vnn.LanguageDecoder
        self.dec = decoder(self.emb_word, 2*args.dhid, 
                           attn_dropout=args.attn_dropout,
                           hstate_dropout=args.hstate_dropout,
                           word_dropout=args.word_dropout,
                           input_dropout=args.input_dropout,
                           train_teacher_forcing=args.train_teacher_forcing,
                           train_student_forcing_prob=args.train_student_forcing_prob)

        # dropouts
        self.vis_dropout = nn.Dropout(args.vis_dropout)
        self.act_dropout = nn.Dropout(args.act_dropout, inplace=True)

        # TODO need to do per subgoal version. ??
        # internal states
        self.state_t = None
        self.e_t = None
        self.test_mode = False

        # paths
        self.root_path = os.getcwd()
        self.feat_pt = 'feat_conv.pt'

        # params
        self.max_subgoals = 25

        # reset model
        self.reset()
        
    def featurize(self, batch):
        '''tensoroze and pad batch input'''
    
        # time
        time_report = defaultdict(int)

        device = torch.device('cuda') if self.args.gpu else torch.device('cpu')

        # feat[feature_name][subgoal index][batch index] = feature iterable over timesteps
        feat = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))
    
        batch_size = len(batch)
        max_num_subgoals = 0
        for batch_i, ex in enumerate(batch):

            # how many subgoals?
            num_subgoals = len(ex['num']['action_low'])
            if num_subgoals > max_num_subgoals:
                max_num_subgoals = num_subgoals     

            #########
            # outputs
            #########

            # time
            start_time = time.time()

            # serialize segments
            # self.serialize_lang_action(ex)
            
            if not self.test_mode: 
                # goal and instr language
                # list(num words in goal), list(num subgoals list(num words in subgoal))
                lang_goal, lang_instr = ex['num']['lang_goal'], ex['num']['lang_instr']
                
                # append instr
                for subgoal_i in range(num_subgoals):
                    feat['lang_instr'][subgoal_i][batch_i] = lang_instr[subgoal_i]

                # # append goal TODO later
                # feat['lang_goal'].append(lang_goal)
   
            # time
            time_report['featurize_outputs'] += time.time() - start_time

            #########
            # inputs
            #########
            # time
            start_time = time.time()
            # low-level action
            for subgoal_i, a_group in enumerate(ex['num']['action_low']):
                feat['action_low'][subgoal_i][batch_i] = [a['action'] for a in a_group]
            # time
            time_report['featurize_input_action_low'] += time.time() - start_time

            # -----------loading image features------------------
            # time
            start_time = time.time()
            
            # load Resnet features from disk
            root = self.get_task_root(ex)

            # time
            torch_load_start_time = time.time()
            # shape (num gold frames for task, 512, 7, 7)
            im = torch.load(os.path.join(root, self.feat_pt))
            # time
            time_report['featurize_torch_load_time'] += time.time() - torch_load_start_time

            # number of low-level actions in trajectory data, excluding <<no-op>> at the end.
            num_low_actions = len(ex['plan']['low_actions'])  # 67
            # number of feat frames exist in raw data for this task
            num_feat_frames = im.shape[0]

            # list len=num subgoals
            num_actions_per_subgoal = [len(a_group) for a_group in  ex['num']['action_low']]

            # list(num subgoals list(time step per subgoal tensor shape (512, 7, 7)))
            keep = [[None for _ in range(num_acts)] for num_acts in num_actions_per_subgoal]
            # last subgoal contains only <<no-op>>
            assert len(keep[-1]) == 1

            if num_low_actions != num_feat_frames:
                last_image_low_idx = 0
                high_idx = 0
                low_idx = 0
                for i, d in enumerate(ex['images']): # 67
                    if ex['images']['high_idx'] > high_idx:
                        low_idx = 0
                        high_idx = ex['images']['high_idx']
                    elif ex['images']['low_idx'] > last_image_low_idx: # 55 > 54
                        low_idx += 1
                    last_image_low_idx = ex['images']['low_idx']
                    # only add frames linked with low-level actions (i.e. skip filler frames like smooth rotations and dish washing)
                    if keep[high_idx][low_idx] is None:
                        keep[high_idx][low_idx] = im[i]
            else:
                high_idx = 0
                low_idx = 0
                for i in range(num_low_actions): # 67
                    if ex['num']['low_to_high_idx'][i] > high_idx:
                        low_idx = 0
                        high_idx = ex['num']['low_to_high_idx'][i]
                    keep[high_idx][low_idx] = im[i]
                    low_idx += 1 
            # append no-op image
            keep[-1][0] = keep[-2][-1]

            for subgoal_i, subgoal_frames in enumerate(keep):
                # tensor shape(num time steps, 512, 7, 7)
                feat['frames'][subgoal_i][batch_i] = torch.stack(subgoal_frames, dim=0)
            
            # time
            time_report['featurize_input_resnet_features'] += time.time() - start_time
            # ------------------------------------------------------

        # time
        start_time = time.time()
        feat['action_low_seq_lengths'] = []
        # tensorization and padding
        for k, v in feat.items():

            if k in {'action_low'}:
                # action embedding and padding
                all_pad_seqs = []
                all_seq_lengths = []
                empty_tensor = torch.ones(torch.tensor(v[0][0][0]).unsqueeze(0).shape, device=device) * self.pad
                for subgoal_i in range(max_num_subgoals):
                    # list of length B. each shaped (l,) with l = time steps in subgoal, value is integer action index.
                    seqs = []
                    seq_lengths = []
                    for batch_i in range(batch_size):
                        if isinstance(v[subgoal_i][batch_i], type(None)):
                            seqs.append(empty_tensor)
                            seq_lengths.append(0) # TODO not sure if this works downstream 
                        else:
                            seqs.append(torch.tensor(v[subgoal_i][batch_i], device=device))
                            seq_lengths.append(len(v[subgoal_i][batch_i]))
                    # tensor shape (B, t) with t = max(l)
                    all_pad_seqs.append(pad_sequence(seqs, batch_first=True, padding_value=self.pad))
                    # (B,). Each value is l= num timesteps for the subgoal.
                    all_seq_lengths.append(np.array(seq_lengths))

                assert all_pad_seqs[-1].shape[0] == all_seq_lengths[-1].shape[0] == batch_size
                # list length=max_num_subgoals, each (B, t) with T = max(l)
                feat[k] = all_pad_seqs
                # list length=max_num_subgoals, each (B,). Each value is l for the example
                feat[k+'_seq_lengths'] = all_seq_lengths

            elif not 'seq_lengths' in k:  # lang_instr, frames

                # default: tensorize and pad sequence
                all_pad_seqs = []
                empty_tensor = torch.ones(torch.tensor(v[0][0][0]).unsqueeze(0).shape, device=device, dtype=torch.float if ('frames' in k) else torch.long) * self.pad
                for subgoal_i in range(max_num_subgoals):
                    # list of length B. each shaped (l, *). l = time steps in subgoal.
                    seqs = []
                    for batch_i in range(batch_size):
                        if isinstance(v[subgoal_i][batch_i], type(None)):
                            seqs.append(empty_tensor)
                        else:
                            seqs.append(torch.tensor(v[subgoal_i][batch_i], device=device, dtype=torch.float if ('frames' in k) else torch.long))
                    # tensor shape (B, t, *) with t = max(l)
                    # tensor shape (B, t, 512, 7, 7) for k='frames'
                    all_pad_seqs.append(pad_sequence(seqs, batch_first=True, padding_value=self.pad))

                # list length=max_num_subgoals, each (B, t, *) with T = max(l)
                # list length=max_num_subgoals, each (B, t, 512, 7, 7) for k='frames'
                assert all_pad_seqs[-1].shape[0] == batch_size
                feat[k] = all_pad_seqs
        # time
        time_report['featurize_tensorization_and_padding'] += time.time() - start_time

        return feat, time_report
    
    # def serialize_lang_action(self, feat):
    #     '''
    #     append segmented instr language and low-level actions into single sequences
    #     '''
    #     is_serialized = not isinstance(feat['num']['lang_instr'][0], list)
    #     if not is_serialized:
    #         feat['num']['action_low'] = [a for a_group in feat['num']['action_low'] for a in a_group]
    #         if not self.test_mode:
    #             feat['num']['lang_instr'] = [word for desc in feat['num']['lang_instr'] for word in desc]

    def forward_one_subgoal(self, feat_subgoal, last_enc_state, last_dec_state, max_decode=300, validate_teacher_forcing=False, validate_sample_output=False):
        '''
        feat_subgoal : feature dictionary returned from self.featurize()
        last_enc_state: encoder (h_0, c_0) from the last subgoal.
        last_dec_state: decoder (h_0, c_0) from the last subgoal.
        max_decode: integer. Max num words to produce for the subgoal.
        validate_teacher_forcing: Boolean. Whether to use teacher forcing when we are in validation mode.
        validate_sample_output: Boolean. Only used when validate_teacher_forcing=False. True will sample output tokens from output distribution. False will use argmax decoding.
        '''
        # encode subgoal action seq and frames
        if last_enc_state is None:
            # encode entire sequence of low-level actions
            cont_act, enc_act, curr_enc_state = self.enc(feat_subgoal)
        else:
            # encode entire sequence of low-level actions
            cont_act, enc_act, curr_enc_state = self.enc(feat_subgoal, last_enc_state)

        # decode subgoal language instruction
        if last_dec_state is None:
            # cont_act has shape (B, *)
            last_dec_state = cont_act, torch.zeros_like(cont_act)

        # Use last_dec_state, enc_act, 

        # run decoder until entire sentence in subgoal is finished
        res, curr_dec_state = self.dec(enc_act, max_decode=max_decode, gold=feat_subgoal['lang_instr'], state_0=last_dec_state, 
        validate_teacher_forcing=validate_teacher_forcing, validate_sample_output=validate_sample_output)

        return res, curr_enc_state, curr_dec_state

    def forward(self, feat, max_decode=300, validate_teacher_forcing=False, validate_sample_output=False):
        '''
        feat : feature dictionary returned from self.featurize()
        max_decode: integer. Max num words to produce for the subgoal.
        validate_teacher_forcing: Boolean. Whether to use teacher forcing when we are in validation mode.
        validate_sample_output: Boolean. Only used when validate_teacher_forcing=False. True will sample output tokens from output distribution. False will use argmax decoding.
        '''
        batch_num_subgoals = len(feat['action_low'])
        batch_size = feat['action_low'][0].shape[0]

        enc_state, dec_state = None, None
        for subgoal_i in range(batch_num_subgoals):
            feat_subgoal = {k:v[subgoal_i] for k,v in feat}
            res_subgoal, enc_state, dec_state = self.forward_one_subgoal(   feat_subgoal, enc_state, dec_state,
                                                                            max_decode=max_decode, 
                                                                            validate_teacher_forcing=validate_teacher_forcing, 
                                                                            validate_sample_output=validate_sample_output)
            for k in res_subgoal.keys():
                feat[k][subgoal_i] = res_subgoal[k]
        return feat
    
    # def reset(self):
    #     '''
    #     reset internal states (used for real-time execution during eval)
    #     '''
    #     self.r_state = {
    #         'state_t': None,
    #         'e_t': None,
    #         'cont_act': None,
    #         'enc_act': None
    #     }
    
    # def step(self, feat, prev_word=None):
    #     '''
    #     forward the model for a single time-step (used for real-time execution during eval)
    #     '''
    #     # encode action features
    #     if self.r_state['cont_act'] is None and self.r_state['enc_act'] is None:
    #         self.r_state['cont_act'], self.r_state['enc_act'] = self.enc(feat)

    #     # initialize embedding and hidden states
    #     if self.r_state['e_t'] is None and self.r_state['state_t'] is None:
    #         self.r_state['e_t'] = self.dec.go.repeat(self.r_state['enc_act'].size(0), 1)
    #         self.r_state['state_t'] = self.r_state['cont_act'], torch.zeros_like(self.r_state['cont_act'])

    #     # previous
    #     e_t = self.embed_lang(prev_word) if prev_word is not None else self.r_state['e_t']

    #     # decode and save embedding and hidden states
    #     out_word_low, state_t, *_ = self.dec.step(self.r_state['enc_act'], e_t=e_t, state_tm1=self.r_state['state_t'])

    #     # save states
    #     self.r_state['state_t'] = state_t
    #     self.r_state['e_t'] = self.dec.emb(out_word_low.max(1)[1])

    #     # output formatting
    #     feat['out_word_low'] = out_word_low.unsqueeze(0)
    #     return feat

    def extract_preds(self, out, batch, feat, clean_special_tokens=True):
        '''
        output processing
        '''
        # batch -- list of loaded tasks 
        # feat['out_lang_instr'] -- list of length num_subgoals, each tensor shaped (B, T, vocab size)
        pred = defaultdict(lambda : defaultdict(lambda : defaultdict(str)))
        for subgoal_i, lang_instr_pred in enumerate(feat['out_lang_instr']):
            for ex, lang_instr in zip(batch, lang_instr_pred.max(2)[1].tolist()):
                # remove padding tokens
                if self.pad in lang_instr:
                    pad_start_idx = lang_instr.index(self.pad)
                    lang_instr = lang_instr[:pad_start_idx]
                
                if clean_special_tokens:
                    if self.stop_token in lang_instr:
                        stop_start_idx = lang_instr.index(self.stop_token)
                        lang_instr = lang_instr[:stop_start_idx]

                # index to word tokens
                words = self.vocab['word'].index2word(lang_instr)

                task_id_ann = self.get_task_and_ann_id(ex)
                pred[task_id_ann]['lang_instr'][subgoal_i] = ' '.join(words)

        # passed to compute_metric eventually
        return pred

    def embed_word(self, lang):
        '''
        embed language
        called only in step -- eval_* modules
        '''
        device = torch.device('cuda') if self.args.gpu else torch.device('cpu')
        lang_num = torch.tensor(self.vocab['word'].word2index(lang), device=device)
        # point back to self.emb_word
        lang_emb = self.dec.emb(lang_num).unsqueeze(0)
        return lang_emb
    
    def compute_loss(self, out, batch, feat):
        '''
        loss function for Seq2Seq agent
        '''
        device = torch.device('cuda') if self.args.gpu else torch.device('cpu')

        losses = dict()
        batch_size, num_subgoals = feat['action_low'][0].shape[0], len(feat['action_low'])
        # shape (B, num_subgoals)
        all_subgoals_per_task_loss = torch.empty(batch_size, num_subgoals, device=device, dtype=torch.float)
        # shape (B, num_subgoals)
        all_subgoals_non_empty = torch.empty(batch_size, num_subgoals, device=device, dtype=torch.float)

        num_subgoals = len(feat['lang_instr'])
        for subgoal_i in range(num_subgoals):
            # GT and predictions
            if self.training:
                # (B*T, Vocab Size), raw unormalized scores
                p_lang_instr = out['out_lang_instr'][subgoal_i].view(-1, len(self.vocab['word']))
            else:
                # Trim prediction to match sequence lengths first
                gold_lang_instr_length = feat['lang_instr'][subgoal_i].shape[1]
                p_lang_instr = out['out_lang_instr'][subgoal_i][:, :gold_lang_instr_length, :].reshape(-1, len(self.vocab['word']))
            l_lang_instr = feat['lang_instr'][subgoal_i].view(-1)

            # language instruction loss
            # tenosr shape (B, t), whether token is valid word
            pad_valid = (l_lang_instr != self.pad).float()
            # tensor shape (B, ), how many tokens are valid words
            num_valid = torch.sum(pad_valid, dim=1)
            # tensor shape (B, ), whether subgoal has any valid words
            bool_valid = (num_valid > 0).float()
            all_subgoals_non_empty[:, subgoal_i] = bool_valid
            # tenosr shape (B, t)
            lang_instr_loss = F.cross_entropy(p_lang_instr, l_lang_instr, reduction='none')
            # tenosr shape (B, t)
            lang_instr_loss *= pad_valid
            # Average across all timesteps in a subgoal, per task
            # tensor shape (B, )
            all_subgoals_per_task_loss[:, subgoal_i] = torch.div(torch.sum(lang_instr_loss, dim=1), num_valid) * bool_valid

        # Average across all subgoals, per task
        # tensor shape (B, ), how many subgoals are valid in a task
        subgoals_valid = torch.sum(all_subgoals_non_empty, dim=1)
        # no task should have less than 2 subgoals (including <<np-op>>)
        assert torch.sum(subgoals_valid < 2) == 0
        # tensor shape (B, )
        all_subgoals_per_task_loss = torch.div(torch.sum(all_subgoals_per_task_loss, dim=1), subgoals_valid)
        # Average across all tasks
        lang_instr_loss = all_subgoals_per_task_loss.mean()

        losses['lang_instr'] = lang_instr_loss
        perplexity = 2**lang_instr_loss

        return losses, perplexity

    def compute_metric(self, preds, data):
        '''
        compute BLEU score for output
        '''

        # how does this work during training with teacher forcing !?
        m = collections.defaultdict(list)

        flatten_isntr = lambda instr: [word.strip() for sent in instr for word in sent]

        all_pred_id_ann = list(preds.keys())
        for task in data:
            # find matching prediction
            pred_id_ann = '{}_{}'.format(task['task'].split('/')[1], task['repeat_idx'])
            # grab task data for ann_0, ann_1 and ann_2
            exs = self.load_task_jsons(task)
            # make sure all human annotations have same number of subgoals
            assert len(set([len(ex['ann']['instr']) for ex in exs])) == 1
            # compute metric for each subgoal
            num_subgoals = len(exs[0]['ann']['instr'])
            bleu_all_subgoals = []
            for subgoal_i in range(num_subgoals):
                # a list of 3 lists of word tokens. (1 for each human annotation, so total 3)
                ref_lang_instrs = [ex['ann']['instr'][subgoal_i] for ex in exs] 
                # compute bleu score for subgoal
                bleu_all_subgoals.append(sentence_bleu(ref_lang_instrs, preds[pred_id_ann]['lang_instr'][subgoal_i].split(' ')))
            # average bleu score across all subgoals
            m['BLEU'].append(sum(bleu_all_subgoals)/num_subgoals)
            all_pred_id_ann.remove(pred_id_ann)

        assert len(all_pred_id_ann) == 0
        return {k: sum(v)/len(v) for k, v in m.items()}