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
import json
from collections import defaultdict

class Module(Base):

    def __init__(self, args, vocab, object_vocab):
        '''
        Seq2Seq agent
        '''
        super().__init__(args, vocab, object_vocab)

        self.pp_folder = args.pp_folder

        self.encoder_addons = encoder_addons
        self.decoder_addons = decoder_addons

        # linear project for instance embeddings
        if self.object_repr == 'instance':
            obj_demb = self.emb_object.weight.size(1)
            self.instance_fc = nn.Linear(obj_demb + obj_demb + 1, args.dhid)
        else:
            self.instance_fc = None

        # action sequence decoder
        if self.encoder_addons == 'max_pool_obj':
            encoder = vnn.ActionFrameAttnEncoderPerSubgoalMaxPool
        elif self.encoder_addons == 'biattn_obj':
            encoder = vnn.ActionFrameAttnEncoderPerSubgoalObjAttn
        else: # 'none'
            encoder = vnn.ActionFrameAttnEncoderPerSubgoal

        self.enc = encoder( emb=self.emb_action_low,
                            obj_emb=self.emb_object,
                            object_repr=self.object_repr,
                            dframe=args.dframe, 
                            dhid=args.dhid,
                            instance_fc=self.instance_fc,
                            act_dropout=args.act_dropout,
                            vis_dropout=args.vis_dropout,
                            input_dropout=args.input_dropout,
                            hstate_dropout=args.hstate_dropout,
                            attn_dropout=args.attn_dropout,
                            bidirectional=True)   

        # language decoder
        if self.decoder_addons == 'aux_loss':
            self.aux_loss_over_object_states = True
        else: # 'none'
            self.aux_loss_over_object_states = False
        
        decoder = vnn.LanguageDecoder
        self.dec = decoder( emb=self.emb_word,
                            obj_emb=self.emb_object,
                            dhid=2*args.dhid,
                            object_repr=self.object_repr,
                            instance_fc=self.instance_fc,
                            attn_dropout=args.attn_dropout,
                            hstate_dropout=args.hstate_dropout,
                            word_dropout=args.word_dropout,
                            input_dropout=args.input_dropout,
                            train_teacher_forcing=args.train_teacher_forcing,
                            train_student_forcing_prob=args.train_student_forcing_prob,
                            aux_loss_over_object_states=self.aux_loss_over_object_states)

        # dropouts
        self.vis_dropout = nn.Dropout(args.vis_dropout)
        self.act_dropout = nn.Dropout(args.act_dropout, inplace=True)

        # TODO need to do per subgoal version. ??
        # internal states
        self.state_t = None
        self.e_t = None
        self.test_mode = False

        # Auxiliary Loss
        self.aux_criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.language_crtierion = nn.CrossEntropyLoss(reduction='mean', ignore_index=0)

        # paths
        self.root_path = os.getcwd()
        self.feat_pt = 'feat_conv.pt'

        # params
        self.max_subgoals = 25

        # reset model
        # self.reset() # TODO do we need to reset?
        
    def featurize(self, batch):
        '''tensoroze and pad batch input'''
    
        # time
        time_report = defaultdict(int)

        device = torch.device('cuda') if self.args.gpu else torch.device('cpu')

        # feat[feature_name][subgoal index][batch index] = feature iterable over timesteps
        feat = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))
    
        batch_size = len(batch)
        max_num_subgoals = 0
        max_num_objects = 0

        # TODO try multiprocess pooling version
        for batch_i, ex in enumerate(batch):

            # ignore last subgoal (i.e. no-op)
            num_subgoals = len(ex['num']['action_low']) - 1
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
                    feat['lang_instr'][subgoal_i][batch_i] = lang_instr[subgoal_i] + [self.word_stop_token]
   
            # time
            time_report['featurize_outputs'] += time.time() - start_time

            #########
            # inputs
            #########

            # -----------loading action tokens------------------
            # time
            start_time = time.time()
            # low-level action
            for subgoal_i in range(num_subgoals):
                a_group = ex['num']['action_low'][subgoal_i]
                feat['action_low'][subgoal_i][batch_i] = [a['action'] for a in a_group] + [self.action_stop_token]
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
            num_low_actions = len(ex['plan']['low_actions'])  # 67, excludes last subgoal <<stop>> 
            # number of feat frames exist in raw data for this task
            num_feat_frames = im.shape[0]  # 67, excludes last subgoal <<stop>> 

            # list len=num subgoals
            num_actions_per_subgoal = [len(ex['num']['action_low'][subgoal_i]) for subgoal_i in range(num_subgoals)]

            # list(num subgoals list(time step per subgoal tensor shape (512, 7, 7)))
            keep = [[None for _ in range(num_acts)] for num_acts in num_actions_per_subgoal]

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

            for subgoal_i, subgoal_frames in enumerate(keep):
                # append no-op image to last of each subgoal
                subgoal_frames.append(subgoal_frames[-1])
                # tensor shape(num time steps, 512, 7, 7)
                feat['frames'][subgoal_i][batch_i] = torch.stack(subgoal_frames, dim=0)
            
            # time
            time_report['featurize_input_resnet_features'] += time.time() - start_time
            # -----------loading state features------------------

            if self.encoder_addons != 'none' and self.decoder_addons != 'none':

                states_root = root.replace('train/', '').replace('valid_seen/', '').replace('valid_unseen/', '')
                with open(os.path.join(states_root, '{}/extracted_feature_states.json'.format(self.pp_folder)), 'r') as f:
                    obj_states = json.load(f)

                num_objects = len(obj_states['objectTypeList'])
                if num_objects > max_num_objects:
                    max_num_objects = num_objects

                for subgoal_i in range(num_subgoals):
                    if self.object_repr == 'type':
                        # each is a list (timestep) of lists (num objects)
                        # [[apple 1, pear 1, ...], [apple 1 , pear 0, ...], [apple 0, pear 1, ...]]
                        feat['object_state_change'][subgoal_i][batch_i] = obj_states['type_state_change'][subgoal_i] + [[0] * num_objects]  # include <<stop>>
                        feat['object_state_change_since_last_subgoal'][subgoal_i][batch_i] = obj_states['type_state_change_since_last_subgoal'][subgoal_i] + [[0] * num_objects]  # include <<stop>>
                        feat['object_visibility'][subgoal_i][batch_i] = obj_states['type_visibile'][subgoal_i] + [obj_states['type_visibile'][subgoal_i][-1]]  # include <<stop>>
                        # [[apple idx, pear idx, ...], [apple idx, pear idx, ...], [apple idx, pear idx, ...]]
                        num_timesteps = len(obj_states['type_state_change'][subgoal_i]) + 1  # include <<stop>>
                        feat['object_token_id'][subgoal_i][batch_i] = [obj_states['objectTypeList_TypeNum'] for _ in range(num_timesteps)]
                    elif self.object_repr == 'instance':
                        # each is a list (timestep) of lists (num objects)
                        # [[apple 1, pear 1, ...], [apple 1 , pear 0, ...], [apple 0, pear 1, ...]]
                        feat['object_state_change'][subgoal_i][batch_i] = obj_states['instance_state_change'][subgoal_i] + [[0] * num_objects]  # include <<stop>>
                        feat['object_state_change_since_last_subgoal'][subgoal_i][batch_i] = obj_states['instance_state_change_since_last_subgoal'][subgoal_i] + [[0] * num_objects]  # include <<stop>>
                        feat['object_visibility'][subgoal_i][batch_i] = obj_states['instance_visibile'][subgoal_i] + [obj_states['instance_visibile'][subgoal_i][-1]]  # include <<stop>>
                        feat['object_receptacle_change'][subgoal_i][batch_i] = obj_states['instance_receptacle_change'][subgoal_i] + [[0] * num_objects]  # include <<stop>>
                        feat['object_receptacle_change_since_last_subgoal'][subgoal_i][batch_i] = obj_states['instance_receptacle_change_since_last_subgoal'][subgoal_i] + [[0] * num_objects]  # include <<stop>>
                        feat['object_distance'][subgoal_i][batch_i] = obj_states['instance_distance'][subgoal_i] + [obj_states['instance_distance'][subgoal_i][-1]]  # include <<stop>>
                        feat['receptacle_token_id'][subgoal_i][batch_i] = obj_states['objectReceptacleList_TypeNum'][subgoal_i] + [obj_states['objectReceptacleList_TypeNum'][subgoal_i][-1]]  # include <<stop>>
                        # [[apple idx, pear idx, ...], [apple idx, pear idx, ...], [apple idx, pear idx, ...]]
                        num_timesteps = len(obj_states['instance_state_change'][subgoal_i]) + 1  # include <<stop>>
                        feat['object_token_id'][subgoal_i][batch_i] = [obj_states['objectInstanceList_TypeNum'] for _ in range(num_timesteps)]                        
                         
        # tensorization and padding
        # time
        start_time = time.time()
        feat['action_low_seq_lengths'] = []
        for k, v in feat.items():

            if k == 'action_low':

                all_pad_seqs = []
                all_seq_lengths = []
                empty_tensor = torch.ones(torch.tensor(v[0][0][0]).unsqueeze(0).shape, device=device, dtype=torch.long) * self.pad

                for subgoal_i in range(max_num_subgoals):
                    # list of length B. each shaped (l,) with l = time steps in subgoal, value is integer action index.
                    seqs = []
                    seq_lengths = []
                    for batch_i in range(batch_size):
                        if isinstance(v[subgoal_i][batch_i], type(None)):
                            seqs.append(empty_tensor)
                            seq_lengths.append(1) # TODO not sure if this works downstream 
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

            elif k in {'lang_instr', 'frames'}:

                all_pad_seqs = []
                # v[0][0][0] -- [subgoal 0][batch 0][time step 0]
                # shape (1, *)
                empty_tensor = torch.ones(torch.tensor(v[0][0][0]).unsqueeze(0).shape, device=device, dtype=torch.float if ('frames' in k) else torch.long) * self.pad

                for subgoal_i in range(max_num_subgoals):
                    # list of length B. each shaped (l, *). l = time steps in subgoal.
                    seqs = []
                    for batch_i in range(batch_size):
                        if isinstance(v[subgoal_i][batch_i], type(None)):
                            # each shape (1, *)
                            seqs.append(empty_tensor)
                        else:
                            # each shape (t, *)
                            seqs.append(torch.tensor(v[subgoal_i][batch_i], device=device, dtype=torch.float if ('frames' in k) else torch.long))
                    # each tensor shape (B, t, *) with t = max(l)
                    # each tensor shape (B, t, 512, 7, 7) for k='frames'
                    all_pad_seqs.append(pad_sequence(seqs, batch_first=True, padding_value=self.pad))

                # list length=max_num_subgoals, each (B, t, *) with T = max(l)
                # list length=max_num_subgoals, each (B, t, 512, 7, 7) for k='frames'
                assert all_pad_seqs[-1].shape[0] == batch_size
                feat[k] = all_pad_seqs
            
            elif k in {
                'object_token_id', 'receptacle_token_id', 
                'object_state_change', 'object_state_change_since_last_subgoal', 
                'object_receptacle_change', 'object_receptacle_change_since_last_subgoal', 
                'object_distance', 'object_visibility'}:

                all_pad_seqs = []
                # shape (1, max_num_objects) -- single time step, 40 objects
                empty_tensor = torch.ones((1, max_num_objects), device=device, dtype=torch.long if ('token_id' in k) else torch.float) * self.pad

                for subgoal_i in range(max_num_subgoals):
                    # list of length B. each shaped (l, *). l = time steps in subgoal.
                    seqs = []
                    for batch_i in range(batch_size):
                        if isinstance(v[subgoal_i][batch_i], type(None)):
                            # each shape (1, max_num_objects)
                            seqs.append(empty_tensor)
                        else:
                            # shape (t, max_num_objects)
                            object_tensor = torch.ones(len(v[subgoal_i][batch_i]), max_num_objects, device=device, dtype=torch.long if ('token_id' in k) else torch.float) * self.pad
                            # fill shape shape (t, <=max_num_objects)
                            object_tensor[:, :len(v[subgoal_i][batch_i][0])] = torch.tensor(v[subgoal_i][batch_i])
                            seqs.append(object_tensor)
                    # each tensor shape (B, t, max_num_objects) with t = max(l)
                    all_pad_seqs.append(pad_sequence(seqs, batch_first=True, padding_value=self.pad))

                # list length=max_num_subgoals, each (B, t, max_num_objects) with T = max(l)
                assert all_pad_seqs[-1].shape[0] == batch_size and all_pad_seqs[-1].shape[2] == max_num_objects
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

    def forward_one_subgoal(self, feat_subgoal, max_decode=300, 
                            validate_teacher_forcing=False, validate_sample_output=False):
        '''
        feat_subgoal : feature dictionary returned from self.featurize()
        max_decode: integer. Max num words to produce for the subgoal.
        validate_teacher_forcing: Boolean. Whether to use teacher forcing when we are in validation mode.
        validate_sample_output: Boolean. Only used when validate_teacher_forcing=False. True will sample output tokens from output distribution. False will use argmax decoding.
        '''

        # encode entire subgoal sequence of low-level actions and frames
        # (B, args.dhid*2), (B, t, args.dhid*2), (h_n, c_n)
        cont_act, enc_act, curr_enc_state = self.enc(
            feat_subgoal=feat_subgoal, 
            last_subgoal_hx=None
            )

        # cont_act has shape (B, args.dhid*2)
        dec_state_0 = cont_act, torch.zeros_like(cont_act)

        # run decoder until entire sentence in subgoal is finished
        res, curr_dec_state = self.dec(
            enc=enc_act,
            feat_subgoal=feat_subgoal,
            max_decode=max_decode,
            state_0=dec_state_0,
            valid_object_indices = feat_subgoal['object_token_id'][:,0,:] if self.aux_loss_over_object_states else None, 
            validate_teacher_forcing=validate_teacher_forcing, 
            validate_sample_output=validate_sample_output,
            )

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

        for subgoal_i in range(batch_num_subgoals):
            feat_subgoal = {k:v[subgoal_i] for k,v in feat.items()}
            # dict, ((B, 2*args.dhid), (B, 2*args.dhid)), 
            res_subgoal, enc_state, dec_state = self.forward_one_subgoal(
                feat_subgoal=feat_subgoal,
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

    @torch.no_grad()
    def extract_preds(self, out, batch, feat, clean_special_tokens=True):
        '''
        output processing
        '''
        # batch -- list of loaded tasks 
        # out['out_lang_instr'] -- list of length num_subgoals, each tensor shaped (B, T, vocab size)

        get_non_zero_elements = lambda t: t[t.nonzero()].squeeze(1)

        pred = defaultdict(lambda : defaultdict(lambda : defaultdict(str)))
        max_num_subgoals = len(out['out_lang_instr'].keys())

        ct_vis_all = 0
        ct_stc_all = 0

        for subgoal_i in range(max_num_subgoals):
            
            ct_vis = 0
            ct_stc = 0

            # Langugage
            # shape (B, t, word vocab size)
            lang_instr_pred = out['out_lang_instr'][subgoal_i]
            for ex, lang_instr in zip(batch, lang_instr_pred.max(2)[1].tolist()):
                # remove padding tokens
                if self.pad in lang_instr:
                    pad_start_idx = lang_instr.index(self.pad)
                    lang_instr = lang_instr[:pad_start_idx]
                
                if clean_special_tokens:
                    if self.word_stop_token in lang_instr:
                        stop_start_idx = lang_instr.index(self.word_stop_token)
                        lang_instr = lang_instr[:stop_start_idx]

                # index to word tokens
                words = self.vocab['word'].index2word(lang_instr)

                task_id_ann = self.get_task_and_ann_id(ex)
                pred[task_id_ann]['lang_instr'][subgoal_i] = ' '.join(words)

            # Aux Loss
            if self.aux_loss_over_object_states:
                # (B, object vocab size)
                obj_vis_pred = out['out_obj_vis_score'][subgoal_i]
                obj_state_change_pred = out['out_state_change_score'][subgoal_i]

                # (B, max_num_objects of batch)
                obj_valid_indices_out = out['valid_object_indices'][subgoal_i]
                # (B, max_num_objects of batch)
                obj_valid_indices_feat = feat['object_token_id'][subgoal_i][:,0,:]
                assert torch.all(obj_valid_indices_out.eq(obj_valid_indices_feat))

                # (B, t, max_num_objects of batch)
                obj_vis_gold = feat['object_visibility'][subgoal_i]
                obj_state_change_gold = feat['object_state_change_since_last_subgoal'][subgoal_i]
                # (B, ) last time step for each task in this subgoal, exclude stop action
                action_low_seq_lengths = feat['action_low_seq_lengths'][subgoal_i] - 2

                for ex, valid_ixs, last_t, p_vis, p_sc, g_vis, g_sc in zip(
                    batch, obj_valid_indices_out, action_low_seq_lengths,
                    obj_vis_pred, obj_state_change_pred,
                    obj_vis_gold, obj_state_change_gold):

                    valid_ixs = valid_ixs[valid_ixs.nonzero()].squeeze(1)

                    task_id_ann = self.get_task_and_ann_id(ex)
                    # (num_objects in task,)
                    valid_ixs = get_non_zero_elements(valid_ixs)
                    pred[task_id_ann]['p_obj_vis'][subgoal_i] = F.sigmoid(p_vis[valid_ixs])
                    pred[task_id_ann]['p_state_change'][subgoal_i] = F.sigmoid(p_sc[valid_ixs])
                    # (num_objects in task,)
                    pred[task_id_ann]['l_obj_vis'][subgoal_i] = g_vis[last_t,:][:valid_ixs.shape[0]]
                    pred[task_id_ann]['l_state_change'][subgoal_i] = g_sc[last_t,:][:valid_ixs.shape[0]]

                    ct_vis += torch.sum(g_vis[last_t,:][:valid_ixs.shape[0]]).cpu().item()
                    ct_stc += torch.sum(g_sc[last_t,:][:valid_ixs.shape[0]]).cpu().item()

                    assert torch.sum(g_vis[last_t,:][valid_ixs.shape[0]:]) == 0 and torch.sum(g_sc[last_t,:][valid_ixs.shape[0]:]) == 0

            ct_vis_all += ct_vis
            ct_stc_all += ct_stc

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

    def compute_lang_instr_loss(self, p_lang_instr, l_lang_instr):
        '''
        compute language CE loss for language instr
        p_lang_instr: shape (B, t_gold, Vocab Size). Please trun predicted language length to gold language length first.
        l_lang_instr: shape (B, t_gold)
        '''
        # scalar, number of valid tasks in this subgoal.
        num_valid_tasks = torch.sum(torch.sum((l_lang_instr != self.pad), dim=1) > 0).float()
        # shape (B*t_gold, Vocab size)
        p_lang_instr = p_lang_instr.reshape(-1, len(self.vocab['word']))
        # shape (B*t_gold)
        l_lang_instr = l_lang_instr.view(-1)
        # scalar, from taking mean
        loss = self.language_crtierion(input=p_lang_instr, target=l_lang_instr)

        # scalar, scalar
        return loss, num_valid_tasks

    def compute_aux_loss(self, valid_object_indices, predicted_scores, targets):
        '''
        compute Aux loss for object visibility or state change
        valid_object_indices: shape (B, max num objects in batch)
        predicted_scores: shape (B, V)
        targets: shape (B, max num objects in batch)
        '''
        
        device = torch.device('cuda') if self.args.gpu else torch.device('cpu')
        assert valid_object_indices.shape == targets.shape

        # scalar, number of valid tasks in this subgoal.
        num_valid_tasks = torch.sum(torch.sum(valid_object_indices, dim=1) > 0).float()

        # (B, V), expand labels to matching shape
        labels_full = torch.zeros_like(predicted_scores, dtype=torch.float, device=device)
        valids_full = torch.zeros_like(predicted_scores, dtype=torch.float, device=device)
        
        for batch_i in range(valid_object_indices.shape[0]):
            for object_j in range(valid_object_indices.shape[1]):
                labels_full[batch_i, valid_object_indices[batch_i, object_j]] = targets[batch_i, object_j]
                valids_full[batch_i, valid_object_indices[batch_i, object_j]] = valid_object_indices[batch_i, object_j] > 0

        # (B, V)
        loss_full = self.aux_criterion(input=predicted_scores, target=labels_full)
        loss_full *= valids_full
        # (B, )
        loss_per_task = torch.div(torch.sum(loss_full, dim=1), torch.max(torch.sum(valids_full, dim=1), torch.tensor([1], dtype=torch.float, device=device)))

        # scalar, scalar
        return torch.sum(loss_per_task), num_valid_tasks

    def compute_loss(self, out, batch, feat):
        '''
        loss function for Seq2Seq agent
        '''
        device = torch.device('cuda') if self.args.gpu else torch.device('cpu')
        losses = dict()
        batch_size, max_num_subgoals = feat['action_low'][0].shape[0], len(feat['action_low'])

        # Language Instr Loss
        # shape (max_num_subgoals, )
        loss_per_subgoal = torch.empty(max_num_subgoals, device=device, dtype=torch.float)
        tasks_per_subgoal = torch.empty(max_num_subgoals, device=device, dtype=torch.float)
        for subgoal_i in range(max_num_subgoals):
            # get groundtruth and predictions
            # (B,), trim prediction to match sequence lengths first
            gold_lang_instr_length = feat['lang_instr'][subgoal_i].shape[1]
            # (B, t_gold, Vocab Size)
            p_lang_instr = out['out_lang_instr'][subgoal_i][:, :gold_lang_instr_length, :]
            # (B, t_gold)
            l_lang_instr = feat['lang_instr'][subgoal_i]
            # compute loss and number of valid tasks in the subgoal
            loss_per_subgoal[subgoal_i], tasks_per_subgoal[subgoal_i] = \
                self.compute_lang_instr_loss(p_lang_instr, l_lang_instr)

        # shape (max_num_subgoals, ), weighted loss across subgoals
        lang_instr_loss = torch.sum(loss_per_subgoal.mul(tasks_per_subgoal))/torch.sum(tasks_per_subgoal)
        losses['lang_instr_ce_loss'] = lang_instr_loss
        perplexity = 2**lang_instr_loss

        # Object Visibility and State Change Aux Loss
        if self.aux_loss_over_object_states:
            # shape (max_num_subgoals, )
            vis_loss_per_subgoal = torch.empty(max_num_subgoals, device=device, dtype=torch.float)
            state_change_loss_per_subgoal = torch.empty(max_num_subgoals, device=device, dtype=torch.float)
            tasks_per_subgoal_aux_loss = torch.empty(max_num_subgoals, device=device, dtype=torch.float)
            for subgoal_i in range(max_num_subgoals):

                # (B, ) last time step for each task in this subgoal, excluding stop action
                action_low_seq_lengths = feat['action_low_seq_lengths'][subgoal_i] - 2

                vis_loss_per_subgoal[subgoal_i], tasks_per_subgoal_vis = self.compute_aux_loss(
                    valid_object_indices=feat['object_token_id'][subgoal_i][:,0,:],
                    predicted_scores=out['out_obj_vis_score'][subgoal_i],
                    targets=feat['object_visibility'][subgoal_i][torch.arange(batch_size),action_low_seq_lengths,:]
                    )
                state_change_loss_per_subgoal[subgoal_i], tasks_per_subgoal_state_change = self.compute_aux_loss(
                    valid_object_indices=feat['object_token_id'][subgoal_i][:,0,:],
                    predicted_scores=out['out_state_change_score'][subgoal_i],
                    targets=feat['object_state_change_since_last_subgoal'][subgoal_i][torch.arange(batch_size),action_low_seq_lengths,:]
                    )
                assert torch.all(tasks_per_subgoal_vis.eq(tasks_per_subgoal_state_change))
                tasks_per_subgoal_aux_loss[subgoal_i] = tasks_per_subgoal_vis
            losses['obj_vis_bce_loss'] = torch.div(torch.sum(vis_loss_per_subgoal), torch.sum(tasks_per_subgoal_aux_loss))
            losses['obj_state_change_bce_loss'] = torch.div(torch.sum(state_change_loss_per_subgoal), torch.sum(tasks_per_subgoal_aux_loss))
            assert torch.all(tasks_per_subgoal.eq(tasks_per_subgoal_aux_loss))

        return losses, perplexity        

    def classify_preds(self, pred, gt, thresh=0.5):
        '''
        pred : shape (num objects in the task,)
        gt   : shape (num objects in the task,)
        '''
        pred_bool = (pred > 0.5).cpu().numpy()
        gt = gt.cpu().numpy()
        acc = pred_bool == gt
        tp = pred_bool * gt
        fp = pred_bool * (gt == 0)
        fn = (pred_bool == 0) * gt
        return acc, tp, fp, fn

    @torch.no_grad()
    def compute_metric(self, preds, data):
        '''
        compute BLEU score for output
        '''
        # how does this work during training with teacher forcing !?
        m = collections.defaultdict(list)

        flatten_isntr = lambda sent: [word.strip() for word in sent]

        all_pred_id_ann = list(preds.keys())

        # Book keeping to compute Precision and Recall
        TP_VIS_ALL = []
        TP_STC_ALL = []
        FP_VIS_ALL = []
        FP_STC_ALL = []
        FN_VIS_ALL = []
        FN_STC_ALL = []

        for task in data:
            
            # BLEU
            # find matching prediction
            pred_id_ann = '{}_{}'.format(task['task'].split('/')[1], task['repeat_idx'])
            # grab task data for ann_0, ann_1 and ann_2
            exs = self.load_task_jsons(task)
            # make sure all human annotations have same number of subgoals
            assert len(set([len(ex['ann']['instr']) for ex in exs])) == 1

            # compute metric for each subgoal
            num_subgoals = len(exs[0]['ann']['instr']) - 1
            bleu_all_subgoals = []
            for subgoal_i in range(num_subgoals):
                # a list of 3 lists of word tokens. (1 for each human annotation, so total 3)
                ref_lang_instrs = [flatten_isntr(ex['ann']['instr'][subgoal_i]) for ex in exs]
                # compute bleu score for subgoal
                bleu_all_subgoals.append(sentence_bleu(ref_lang_instrs, preds[pred_id_ann]['lang_instr'][subgoal_i].split(' ')))

            # average bleu score across all subgoals
            m['BLEU'].append(sum(bleu_all_subgoals)/num_subgoals)

            # AUX LOSS
            if self.aux_loss_over_object_states:

                for subgoal_i in range(num_subgoals):
                    pred_vis = preds[pred_id_ann]['p_obj_vis'][subgoal_i]
                    gt_vis = preds[pred_id_ann]['l_obj_vis'][subgoal_i]
                    pred_stc = preds[pred_id_ann]['p_state_change'][subgoal_i]
                    gt_stc = preds[pred_id_ann]['l_state_change'][subgoal_i]
                    # Accuracy, TP, FP, FN
                    # each array (num objects in task)
                    acc_vis, tp_vis, fp_vis, fn_vis = self.classify_preds(pred_vis, gt_vis)
                    acc_stc, tp_stc, fp_stc, fn_stc = self.classify_preds(pred_stc, gt_stc)

                    m['ACC_VIS'].extend(acc_vis)
                    m['ACC_STC'].extend(acc_stc)
                    TP_VIS_ALL.extend(tp_vis)
                    TP_STC_ALL.extend(tp_stc)
                    FP_VIS_ALL.extend(fp_vis)
                    FP_STC_ALL.extend(fp_stc)
                    FN_VIS_ALL.extend(fn_vis)
                    FN_STC_ALL.extend(fn_stc)

            all_pred_id_ann.remove(pred_id_ann)

        assert len(all_pred_id_ann) == 0
        m_out = {k: sum(v)/len(v) for k, v in m.items()}

        if self.aux_loss_over_object_states:
            m_out['PRECISION_VIS'] = sum(TP_VIS_ALL) / (sum(TP_VIS_ALL) + sum(FP_VIS_ALL))
            m_out['RECALL_VIS'] = sum(TP_VIS_ALL) / (sum(TP_VIS_ALL) + sum(FN_VIS_ALL))
            m_out['PRECISION_STC'] = sum(TP_STC_ALL) / (sum(TP_STC_ALL) + sum(FP_STC_ALL))
            m_out['RECALL_STC'] = sum(TP_STC_ALL) / (sum(TP_STC_ALL) + sum(FN_STC_ALL))

            m_out['TOTAL_GT_STC'] = sum(TP_STC_ALL) + sum(FN_STC_ALL)
            m_out['TOTAL_PRED_STC'] = sum(TP_STC_ALL) + sum(FP_STC_ALL)
            m_out['TOTAL_GT_VIS'] = sum(TP_VIS_ALL) + sum(FN_VIS_ALL)
            m_out['TOTAL_PRED_VIS'] = sum(TP_VIS_ALL) + sum(FP_VIS_ALL)
            
            m_out['TOTAL_COUNT_VIS'] = len(TP_VIS_ALL)
            m_out['TOTAL_COUNT_STC'] = len(TP_STC_ALL)
            
        return m_out
