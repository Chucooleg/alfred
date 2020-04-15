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

class Module(Base):

    def __init__(self, args, vocab):
        '''
        Seq2Seq agent
        '''
        super().__init__(args, vocab)

        # encoder and self-attention
        self.enc = nn.LSTM(args.demb, args.dhid, bidirectional=True, batch_first=True)
        self.enc_att = vnn.SelfAttn(args.dhid*2)

        # language decoder
        decoder = vnn.LanguageDecoder
        self.dec = decoder(self.emb_word, 2*args.dhid, 
                           attn_dropout=args.attn_dropout,
                           hstate_dropout=args.hstate_dropout,
                           word_dropout=args.word_dropout,
                           input_dropout=args.input_dropout,
                           teacher_forcing=args.dec_teacher_forcing)

        # dropouts
        self.act_dropout = nn.Dropout(args.act_dropout, inplace=True)

        # internal states
        self.state_t = None
        self.e_t = None
        self.test_mode = False

        # paths
        self.root_path = os.getcwd()

        # params
        self.max_subgoals = 25

        # reset model
        self.reset()
        
    def featurize(self, batch):
        '''tensoroze and pad batch input'''
    
        device = torch.device('cuda') if self.args.gpu else torch.device('cpu')
        feat = collections.defaultdict(list)
    
        for ex in batch:
            
            #########
            # outputs
            #########

            # serialize segments
            self.serialize_lang_action(ex)
            
            if not self.test_mode: 
                # goal and instr language
                lang_goal, lang_instr = ex['num']['lang_goal'], ex['num']['lang_instr']
                
                # zero inputs if specified
                lang_goal = self.zero_input(lang_goal) if self.args.zero_goal else lang_goal
                lang_instr = self.zero_input(lang_instr) if self.args.zero_instr else lang_instr
                
                # # append goal
                # feat['lang_goal'].append(lang_goal)
                feat['lang_instr'].append(lang_goal)
                
                # append instr
                # feat['lang_instr'].append(lang_instr)
                
                # append goal + instr
                # lang_goal_instr = lang_goal + lang_instr
                # feat['lang_goal_instr'].append(lang_goal_instr)
            
            #########
            # inputs
            #########
            
            # low-level action
            feat['action_low'].append([a['action'] for a in ex['num']['action_low']])

        # tensorization and padding
        for k, v in feat.items():
            # input
            if k in {'action_low'}:
                # action embedding and padding
                # list of length B. each shaped (L,), value is integer action index.
                seqs = [torch.tensor(vv, device=device) for vv in v]
                # (B, T), T = max(L)
                pad_seq = pad_sequence(seqs, batch_first=True, padding_value=self.pad)
                # (B,). Each value is L for the example
                seq_lengths = np.array(list(map(len, v)))
                # (B, T, args.demb)
                embed_seq = self.emb_action_low(pad_seq)
                # PackedSequence object ready for RNN
                packed_input = pack_padded_sequence(embed_seq, seq_lengths, batch_first=True, enforce_sorted=False)
                feat[k] = packed_input
            else:
                # default: tensorize and pad sequence
                # list of length B. each shaped (L,), value is integer action index.
                seqs = [torch.tensor(vv, device=device, dtype=torch.float if ('frames' in k) else torch.long) for vv in v]
                # (B, T), T = max(L)
                pad_seq = pad_sequence(seqs, batch_first=True, padding_value=self.pad)
                feat[k] = pad_seq     
    
        return feat
    
    def serialize_lang_action(self, feat):
        '''
        append segmented instr language and low-level actions into single sequences
        '''
        is_serialized = not isinstance(feat['num']['lang_instr'][0], list)
        if not is_serialized:
            feat['num']['action_low'] = [a for a_group in feat['num']['action_low'] for a in a_group]
            if not self.test_mode:
                feat['num']['lang_instr'] = [word for desc in feat['num']['lang_instr'] for word in desc]

    def forward(self, feat, max_decode=300):
        # encode entire sequence of low-level actions
        cont_act, enc_act = self.encode_act(feat)
        # run decoder until entire sentence is finished
        state_0 = cont_act, torch.zeros_like(cont_act)
        res = self.dec(enc_act, max_decode=max_decode, gold=feat['lang_instr'], state_0=state_0)
        feat.update(res)
        return feat
    
    def encode_act(self, feat):
        '''
        encode low-level actions
        ''' 
        # PackedSequence object
        emb_act = feat['action_low']
        self.act_dropout(emb_act.data)
        # call nn.LSTM to encode entire sequence
        # PackedSequence obj
        enc_act, _ = self.enc(emb_act)
        # (B, T, args.dhid)
        enc_act, _ = pad_packed_sequence(enc_act, batch_first=True)
        self.act_dropout(enc_act)
        # simple learned self-attention (class SelfAttn)
        cont_act = self.enc_att(enc_act)
        # return both continuous and per-word representations
        return cont_act, enc_act
    
    def reset(self):
        '''
        reset internal states (used for real-time execution during eval)
        '''
        self.r_state = {
            'state_t': None,
            'e_t': None,
            'cont_act': None,
            'enc_act': None
        }
    
    def step(self, feat, prev_word=None):
        '''
        forward the model for a single time-step (used for real-time execution during eval)
        '''

        # encode action features
        if self.r_state['cont_act'] is None and self.r_state['enc_act'] is None:
            self.r_state['cont_act'], self.r_state['enc_act'] = self.encode_act(feat)

        # initialize embedding and hidden states
        if self.r_state['e_t'] is None and self.r_state['state_t'] is None:
            self.r_state['e_t'] = self.dec.go.repeat(self.r_state['enc_act'].size(0), 1)
            self.r_state['state_t'] = self.r_state['cont_act'], torch.zeros_like(self.r_state['cont_act'])

        # previous
        e_t = self.embed_lang(prev_word) if prev_word is not None else self.r_state['e_t']

        # decode and save embedding and hidden states
        out_word_low, state_t, *_ = self.dec.step(self.r_state['enc_act'], e_t=e_t, state_tm1=self.r_state['state_t'])

        # save states
        self.r_state['state_t'] = state_t
        self.r_state['e_t'] = self.dec.emb(out_word_low.max(1)[1])

        # output formatting
        feat['out_word_low'] = out_word_low.unsqueeze(0)
        return feat

    def extract_preds(self, out, batch, feat, clean_special_tokens=True):
        '''
        output processing
        '''
        pred = {}
        #  feat['out_lang_instr'] has shape (B, T, vocab size)
        for ex, lang_instr in zip(batch, feat['out_lang_instr'].max(2)[1].tolist()):
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
            pred[task_id_ann] = {
                'lang_instr': ' '.join(words)
            }
        
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
        losses = dict()

        # GT and predictions
        if self.training:
            # (B*T, Vocab Size), raw unormalized scores
            p_lang_instr = out['out_lang_instr'].view(-1, len(self.vocab['word']))
        else:
            # Trim prediction to match sequence lengths first
            gold_lang_instr_length = feat['lang_instr'].shape[1]
            p_lang_instr = out['out_lang_instr'][:, :gold_lang_instr_length, :].reshape(-1, len(self.vocab['word']))
        l_lang_instr = feat['lang_instr'].view(-1)

        # language instruction loss
        pad_valid = (l_lang_instr != self.pad)
        lang_instr_loss = F.cross_entropy(p_lang_instr, l_lang_instr, reduction='none')
        lang_instr_loss *= pad_valid.float()
        lang_instr_loss = lang_instr_loss.mean()
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
            # a list of 3 lists of word tokens. (1 for each human annotation, so total 3)
            ref_lang_instrs = [flatten_isntr(ex['ann']['instr']) for ex in exs]            
            # compute bleu score
            m['BLEU'].append(sentence_bleu(ref_lang_instrs, preds[pred_id_ann]['lang_instr'].split(' ')))
            all_pred_id_ann.remove(pred_id_ann)
        assert len(all_pred_id_ann) == 0
        return {k: sum(v)/len(v) for k, v in m.items()}
