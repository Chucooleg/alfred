import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np

class SelfAttn(nn.Module):
    '''
    self-attention with learnable parameters
    make a single continuous representation for a sentence.
    '''

    def __init__(self, dhid):
        super().__init__()
        self.scorer = nn.Linear(dhid, 1)

    def forward(self, inp):
        '''
        inp: shape (B, T, *)
        '''
        # shape (B, T, 1), per word score normalized across each sentence
        scores = F.softmax(self.scorer(inp), dim=1)
        # shape (B, 1, T) bmm shape (B, T, *)
        # = shape (B, 1, *).squeeze(1) = shape (B, *)
        cont = scores.transpose(1, 2).bmm(inp).squeeze(1)
        return cont


class DotAttn(nn.Module):
    '''
    dot-attention (or soft-attention)
    '''

    def forward(self, inp, h):
        '''
        inp: shape (B, t, 2*args.dhid)
        h: (B, args.2*dhid)
        '''
        # (B, t, 1)
        score = self.softmax(inp, h)
        # (B, t, 2*args.dhid) element multiply (B, t, 2*args.dhid) = (B, t, 2*args.dhid)
        # sum(B, t, 2*args.dhid, dim=1) = (B, 2*args.dhid, dim=1)
        # (B, 2*args.dhid, dim=1), (B, t, 1)
        return score.expand_as(inp).mul(inp).sum(1), score

    def softmax(self, inp, h):
        # (B, t, 2*args.dhid) mm (B, args.2*dhid, 1) = (B, t, 1)
        raw_score = inp.bmm(h.unsqueeze(2))
        # (B, t, 1)
        score = F.softmax(raw_score, dim=1)
        return score


class ResnetVisualEncoder(nn.Module):
    '''
    visual encoder
    '''

    def __init__(self, dframe):
        super(ResnetVisualEncoder, self).__init__()
        self.dframe = dframe
        self.flattened_size = 64*7*7

        self.conv1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)
        self.fc = nn.Linear(self.flattened_size, self.dframe)
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(64)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))

        x = self.conv2(x)
        x = F.relu(self.bn2(x))

        x = x.view(-1, self.flattened_size)
        x = self.fc(x)

        return x


class ActionFrameAttnEncoder(nn.Module):
    '''
    action and frame sequence encoder base
    '''

    def __init__(self, emb, dframe, dhid,
                 act_dropout=0., vis_dropout=0., bidirectional=True, gpu=False):
        super(ActionFrameAttnEncoder, self).__init__()

        # Embedding matrix for Actions
        self.emb = emb
        self.dhid = dhid
        self.bidirectional = bidirectional
        self.dframe = dframe
        self.demb = emb.weight.size(1)

        # Dropouts
        self.vis_dropout = nn.Dropout(vis_dropout)
        self.act_dropout = nn.Dropout(act_dropout, inplace=True)

        # Image Frame encoder
        self.vis_encoder = ResnetVisualEncoder(dframe=dframe)

        # Self Attn 
        self.enc_att = SelfAttn(dhid*2)

    def vis_enc_step(self, frames):
        '''
        encode image frames for all time steps
        '''        
        B = frames.shape[0]
        T = frames.shape[1]
        # (B*T, 512, 7, 7)
        frames = frames.reshape(B*T, frames.shape[2], frames.shape[3], frames.shape[4])
        vis_feat = self.vis_encoder(frames)
        vis_feat = vis_feat.reshape(B, T, self.dframe)
        return vis_feat

class ActionFrameAttnEncoderFullSeq(ActionFrameAttnEncoder):
    '''
    action and frame sequence encoder. encode all subgoals at once.
    '''

    def __init__(self, emb, dframe, dhid,
                 act_dropout=0., vis_dropout=0., bidirectional=True):
        super(ActionFrameAttnEncoderFullSeq, self).__init__(emb, dframe, dhid,
                 act_dropout=act_dropout, vis_dropout=vis_dropout, bidirectional=bidirectional)

        # Image + Action encoder
        self.encoder = nn.LSTM(self.demb+self.dframe, self.dhid, 
                               bidirectional=self.bidirectional, 
                               batch_first=True)

    def forward(self, feat):
        '''
        encode low-level actions and image frames
        '''
        # Action Sequence
        # (B, T) with T = max(L), already padded
        pad_seq = feat['action_low']
        # (B,). Each value is L for the example
        seq_lengths = feat['action_low_seq_lengths']
        # (B, T, args.demb)
        emb_act = self.emb(pad_seq)

        # Image Frames
        # (B, T, 512, 7, 7) with T = max(L), already padded
        frames = self.vis_dropout(feat['frames'])
        # (B, T, args.dframe)
        vis_feat = self.vis_enc_step(frames)

        # Pack inputs together
        # (B, T, args.demb+args.dframe)
        inp_seq = torch.cat([emb_act, vis_feat], dim=2)
        packed_input = pack_padded_sequence(inp_seq, seq_lengths, batch_first=True, enforce_sorted=False)

        # Encode entire sequence
        # packed sequence object
        enc_act, _ = self.encoder(packed_input)
        # (B, T, args.dhid*2) -- *2 for birdirectional
        enc_act, _ = pad_packed_sequence(enc_act, batch_first=True)
        self.act_dropout(enc_act)

        # Apply learned self-attention
        # (B, args.dhid*2)
        cont_act = self.enc_att(enc_act)

        # return both compact and per-step representations
        return cont_act, enc_act
        

class ActionFrameAttnEncoderPerSubgoal(ActionFrameAttnEncoder):
    '''
    action and frame sequence encoder. encode one subgoal at a time.
    '''

    def __init__(self, emb, obj_emb, object_repr, dframe, dhid, instance_fc,
                act_dropout=0., vis_dropout=0., input_dropout=0., hstate_dropout=0., attn_dropout=0., bidirectional=True):

        super(ActionFrameAttnEncoderPerSubgoal, self).__init__(emb=emb, dframe=dframe, dhid=dhid,
                 act_dropout=act_dropout, vis_dropout=vis_dropout, bidirectional=bidirectional)
            
        # Image + Action encoder
        # action embedding + image vector dim
        # or action embedding + image vector dim
        self.encoder = nn.LSTM( self.demb+self.dframe,
                                self.dhid, 
                                bidirectional=self.bidirectional, 
                                batch_first=True)
        self.input_dropout = nn.Dropout(input_dropout)
        self.hstate_dropout = nn.Dropout(hstate_dropout)
        self.attn_dropout = nn.Dropout(attn_dropout)

        # object handling
        # (V, dhid)
        self.obj_emb = obj_emb
        self.obj_demb = self.obj_emb.weight.size(1)
        self.object_repr = object_repr # 'type' or 'instance

        if self.object_repr == 'instance':
            assert instance_fc is not None
            self.instance_fc = instance_fc

    def make_instance_embeddings(self, object_indices, receptacle_indices, object_distances):
        '''
        object_indices: (B, t, max_num_objects in batch)
        receptacle_indices: (B, t, max_num_objects in batch)
        object_distance: (B, t, max_num_objects in batch)
        '''
        # concat and transform
        
        # (B, t, max_num_objects in batch, obj demb)
        obj_embeddings = self.obj_emb(object_indices)      
        # (B, t, max_num_objects in batch, obj demb)
        recep_embeddings = self.obj_emb(receptacle_indices)
        # (B, t, max_num_objects in batch, obj demb + obj demb + 1)
        cat_embeddings = torch.cat([obj_embeddings, recep_embeddings, object_distances.unsqueeze(-1)], dim=len(obj_embeddings.shape) - 1)
        # (B, t, max_num_objects in batch, dhid)
        return self.instance_fc(cat_embeddings)

    def forward(self, feat_subgoal, last_subgoal_hx):
        '''
        encode low-level actions and image frames

        Args:
        last_subgoal_hx: tuple. (h_0, c_0) argument per nn.LSTM documentation.
        '''
        # Action Sequence
        # (B, t) with t = max(l), already padded
        pad_seq = feat_subgoal['action_low']
        # (B,). Each value is l for the example
        seq_lengths = feat_subgoal['action_low_seq_lengths']
        # (B, t, args.demb)
        emb_act = self.emb(pad_seq)        

        # Image Frames
        # (B, t, 512, 7, 7) with t = max(l), already padded
        frames = self.vis_dropout(feat_subgoal['frames'])
        # (B, t, args.dframe)
        vis_feat = self.vis_enc_step(frames)

        # ( B, t, args.demb+args.dframe)
        inp_seq = torch.cat([emb_act, vis_feat], dim=2)
        packed_input = pack_padded_sequence(inp_seq, seq_lengths, batch_first=True, enforce_sorted=False)

        # Encode entire subgoal sequence
        # packed sequence object, (h_n, c_n) tuple
        enc_act, curr_subgoal_states = self.encoder(input=packed_input, hx=last_subgoal_hx)
        # (B, t, args.dhid*2)
        enc_act, _ = pad_packed_sequence(enc_act, batch_first=True)
        self.act_dropout(enc_act)

        # Apply learned self-attention
        # (B, args.dhid*2)
        cont_act = self.enc_att(enc_act)

        # return both compact, per-step representations, (h_n, c_n) for starting next subgoal encoding
        return cont_act, enc_act, curr_subgoal_states


class ActionFrameAttnEncoderPerSubgoalObjAttn(ActionFrameAttnEncoderPerSubgoal):
    '''
    action and frame sequence encoder. encode one subgoal at a time. use attention over object states
    '''

    def __init__(self, emb, obj_emb, object_repr, dframe, dhid, instance_fc,
                 act_dropout=0., vis_dropout=0., input_dropout=0., hstate_dropout=0., attn_dropout=0., bidirectional=True):

        super(ActionFrameAttnEncoderPerSubgoalObjAttn, self).__init__(emb=emb, obj_emb=obj_emb, object_repr=object_repr, dframe=dframe, dhid=dhid, instance_fc=instance_fc,
                act_dropout=act_dropout, vis_dropout=vis_dropout, input_dropout=input_dropout ,hstate_dropout=hstate_dropout, attn_dropout=attn_dropout, bidirectional=bidirectional)

        # object states handling
        # dhid to dhid
        self.obj_attn = DotAttn()
        # linear for hidden state before attention
        self.h_tm1_fc = nn.Linear(dhid, dhid)

        # Image + Action + obj attn encoder
        # action embedding + image vector dim + obj embed + obj embed
        self.forward_cell = nn.LSTMCell(self.demb+self.dframe+self.dhid+self.dhid, self.dhid)
        if bidirectional:
            self.backward_cell = nn.LSTMCell(self.demb+self.dframe+self.dhid+self.dhid, self.dhid)

    def filter_obj_embeds(self, object_indices, object_boolean_features, receptacle_indices=None, object_distances=None):
        '''
        make continuous repr for object embeddding, visibility and state change.
        '''
        if self.object_repr == 'instance':
            # (B, t, max_num_objects in batch, dhid)
            obj_embeddings = self.make_instance_embeddings(object_indices, receptacle_indices, object_distances)            
        else: # 'type'
            # (B, t, max_num_objects in batch, dhid)
            obj_embeddings = self.obj_emb(object_indices)

        # (B, t, max_num_objects in batch, dhid)
        obj_embeddings = object_boolean_features.unsqueeze(-1).expand_as(obj_embeddings).mul(obj_embeddings)
        return obj_embeddings

    def step(self, cell, emb_act_t, vis_feat_t, obj_vis_t, obj_stc_t, state_tm1):
        # previous decoder hidden state
        # (B, args.dhid)
        h_tm1 = state_tm1[0]

        # inputs (B, V, args.dhid), (B, args.dhid)
        # output (B, args.dhid), (B, t, 1)
        weighted_obj_vis_t, obj_vis_attn_t = self.obj_attn(self.attn_dropout(obj_vis_t), self.h_tm1_fc(h_tm1))
        weighted_obj_stc_t, obj_stc_attn_t = self.obj_attn(self.attn_dropout(obj_stc_t), self.h_tm1_fc(h_tm1))

        # action embedding + image vector dim + obj vis embed + obj stc embed
        inp_t = torch.cat([emb_act_t, vis_feat_t, weighted_obj_vis_t, weighted_obj_stc_t], dim=1)
        inp_t = self.input_dropout(inp_t)

        # update hidden state
        # (B, args.dhid, ), (B, args.dhid, )
        state_t = cell(inp_t, state_tm1)
        state_t = [self.hstate_dropout(x) for x in state_t]
        
        return state_t, obj_vis_attn_t, obj_stc_attn_t

    def encode_sequence_one_direction(self, emb_act, vis_feat, obj_vis, obj_stc, last_subgoal_hx, backward=False):
        '''
        encode entire sequence of inputs in a single direction
        emb_act:   (B, t, args.demb)
        vis_feat:  (B, t, args.dframe)
        obj_vis:   (B, t, max_num_objects in batch, demb)
        obj_stc:   (B, t, max_num_objects in batch, demb)
        last_subgoal_hx: tuple. (h_0, c_0) argument per nn.LSTM documentation. This has shape =((B, args.dhid), (B, args.dhid)).
        '''
        assert last_subgoal_hx[0].shape == last_subgoal_hx[1].shape == (emb_act.shape[0], self.dhid)

        max_t = emb_act.shape[1]
        timesteps = range(max_t) if not backward else range(max_t)[::-1]
        cell = self.forward_cell if not backward else self.backward_cell
        
        state_t = last_subgoal_hx
        assert last_subgoal_hx[0].shape == last_subgoal_hx[1].shape == (emb_act.shape[0], self.dhid)

        # (B, t, args.dhid)
        enc_act = torch.zeros(emb_act.shape[0], max_t, self.dhid, device=emb_act.device, dtype=torch.float)
        # obj_vis_attn_scores, obj_stc_attn_scores = [], []
        for t in timesteps:
            state_t, obj_vis_attn_t, obj_stc_attn_t = self.step(cell, emb_act[:,t,:], vis_feat[:,t,:], obj_vis[:,t,:], obj_stc[:,t,:], state_t)
            enc_act[:,t,:] = state_t[0]
        # (h_n, c_n)=((B, args.dhid), (B, args.dhid)) tuple
        curr_subgoal_states = state_t
        # (B, t, args.dhid), (h_n, c_n)=((B, args.dhid), (B, args.dhid)) tuple
        return enc_act, curr_subgoal_states
            
    def encode_sequence(self, emb_act, vis_feat, obj_vis, obj_stc, last_subgoal_hx):
        '''
        encode entire sequence of inputs
        emb_act:   (B, t, args.demb)
        vis_feat:  (B, t, args.dframe)
        obj_vis:   (B, t, max_num_objects in batch, demb)
        obj_stc:   (B, t, max_num_objects in batch, demb)
        last_subgoal_hx: tuple. (h_0, c_0) argument per nn.LSTM documentation. For bidirectional, this has shape =((B, args.dhid*2), (B, args.dhid*2)).
        '''
        assert emb_act.shape[1] == vis_feat.shape[1] == obj_vis.shape[1] == obj_stc.shape[1]

        if not self.bidirectional:
            # (B, t, args.dhid), (h_n, c_n) tuple
            enc_act, curr_subgoal_states = self.encode_sequence_one_direction(emb_act, vis_feat, obj_vis, obj_stc, last_subgoal_hx)
            # (B, t, args.dhid), (h_n, c_n)=((B, args.dhid), (B, args.dhid)) tuple
            return enc_act, curr_subgoal_states
        else:
            # (B, t, args.dhid), (h_n, c_n) tuple
            enc_act_forward, curr_subgoal_states_forward = self.encode_sequence_one_direction(emb_act, vis_feat, obj_vis, obj_stc, (last_subgoal_hx[0][:,:self.dhid], last_subgoal_hx[1][:,:self.dhid]))
            # (B, t, args.dhid), (h_n, c_n) tuple
            enc_act_backward, curr_subgoal_states_backward = self.encode_sequence_one_direction(emb_act, vis_feat, obj_vis, obj_stc, (last_subgoal_hx[0][:,self.dhid:], last_subgoal_hx[1][:,self.dhid:]), backward=True)
            # (B, t, args.dhid*2)
            enc_act = torch.cat([enc_act_forward, enc_act_backward], dim=2)
            # ((B, args.dhid*2), (B, args.dhid*2))
            curr_subgoal_states = (
                torch.cat([curr_subgoal_states_forward[0], curr_subgoal_states_backward[0]], dim=1), 
                torch.cat([curr_subgoal_states_forward[1], curr_subgoal_states_backward[1]], dim=1)
                )
            # (B, t, args.dhid*2), (h_n, c_n)=((B, args.dhid*2), (B, args.dhid*2)) tuple
            return enc_act, curr_subgoal_states

    def forward(self, feat_subgoal, last_subgoal_hx):
        '''
        encode low-level actions and image frames

        Args:
        last_subgoal_hx: tuple. (h_0, c_0) argument per nn.LSTM documentation. For bidirectional, this has shape =((B, args.dhid*2), (B, args.dhid*2)).
        '''

        # Action Sequence
        # (B, t) with t = max(l), already padded
        pad_seq = feat_subgoal['action_low']
        # (B,). Each value is l for the example
        seq_lengths = feat_subgoal['action_low_seq_lengths']
        # (B, t, args.demb)
        emb_act = self.emb(pad_seq)

        # Image Frames
        # (B, t, 512, 7, 7) with t = max(l), already padded
        frames = self.vis_dropout(feat_subgoal['frames'])
        # (B, t, args.dframe)
        vis_feat = self.vis_enc_step(frames)

        # Make object representation
        if self.object_repr == 'instance':
            # (B, t, max_num_objects in batch, dhid)
            obj_vis = self.filter_obj_embeds(
                feat_subgoal['object_token_id'], feat_subgoal['object_visibility'], 
                receptacle_indices=feat_subgoal['receptacle_token_id'], object_distances=feat_subgoal['object_distance'])
            obj_stc = self.filter_obj_embeds(
                feat_subgoal['object_token_id'], feat_subgoal['object_state_change'], 
                receptacle_indices=feat_subgoal['receptacle_token_id'], object_distances=feat_subgoal['object_distance'])
        else: # 'type'
            # (B, t, max_num_objects in batch, dhid)
            obj_vis = self.filter_obj_embeds(feat_subgoal['object_token_id'], feat_subgoal['object_visibility'])
            obj_stc = self.filter_obj_embeds(feat_subgoal['object_token_id'], feat_subgoal['object_state_change'])

        # hidden state at timestep 0
        if last_subgoal_hx is None:
            h_0 = torch.zeros(emb_act.shape[0], 2*self.dhid if self.bidirectional else self.dhid, dtype=torch.float, device=pad_seq.device)
            c_0 = torch.zeros_like(h_0)
            last_subgoal_hx = (h_0, c_0)

        # (B, t, args.dhid), (h_n, c_n) tuple
        enc_act, curr_subgoal_states = self.encode_sequence(emb_act, vis_feat, obj_vis, obj_stc, last_subgoal_hx)
        self.act_dropout(enc_act)

        # Apply learned self-attention
        # (B, args.dhid*2)
        cont_act = self.enc_att(enc_act)
        
        # (B, args.dhid*2), (B, t, args.dhid*2), (h_n, c_n)
        return cont_act, enc_act, curr_subgoal_states  


class ActionFrameAttnEncoderPerSubgoalMaxPool(ActionFrameAttnEncoderPerSubgoal):
    '''
    action and frame sequence encoder. encode one subgoal at a time. max-pool over object states
    '''

    def __init__(self, emb, obj_emb, object_repr, dframe, dhid, instance_fc,
                 act_dropout=0., vis_dropout=0., input_dropout=0., hstate_dropout=0., attn_dropout=0., bidirectional=True):

        super(ActionFrameAttnEncoderPerSubgoalMaxPool, self).__init__(emb=emb, obj_emb=obj_emb, object_repr=object_repr, dframe=dframe, dhid=dhid, instance_fc=instance_fc,
                act_dropout=act_dropout, vis_dropout=vis_dropout, input_dropout=input_dropout, hstate_dropout=hstate_dropout, attn_dropout=attn_dropout, bidirectional=bidirectional)
            
        # Image + Action encoder
        # action embedding + image vector dim + obj embedding + obj embedding
        # or action embedding + image vector dim
        self.encoder = nn.LSTM( self.demb+self.dframe+self.dhid+self.dhid ,
                                self.dhid, 
                                bidirectional=self.bidirectional, 
                                batch_first=True)       

    def max_pool_object_features(self, object_indices, object_boolean_features, receptacle_indices=None, object_distances=None):
        '''
        Max pool each embedding val across objects.
        object_indices : shape (B, t, max_num_objects in batch). each int index for object vocab.
        object_boolean_features : shape (B, t, max_num_objects in batch). each 1/0
        '''
        if self.object_repr == 'instance':
            # (B, t, max_num_objects in batch, dhid)
            obj_embeddings = self.make_instance_embeddings(object_indices, receptacle_indices, object_distances)               
        else: # 'type'
            # (B, t, max_num_objects in batch, dhid)
            obj_embeddings = self.obj_emb(object_indices)
        # (B, t, max_num_objects in batch, dhid)
        obj_embeddings = object_boolean_features.unsqueeze(-1).expand_as(obj_embeddings).mul(obj_embeddings)
        # (B, t, dhid)
        return obj_embeddings.max(2)[0]

    def forward(self, feat_subgoal, last_subgoal_hx):
        '''
        encode low-level actions and image frames

        Args:
        last_subgoal_hx: tuple. (h_0, c_0) argument per nn.LSTM documentation.
        '''
        # Action Sequence
        # (B, t) with t = max(l), already padded
        pad_seq = feat_subgoal['action_low']
        # (B,). Each value is l for the example
        seq_lengths = feat_subgoal['action_low_seq_lengths']
        # (B, t, args.demb)
        emb_act = self.emb(pad_seq)        

        # Image Frames
        # (B, t, 512, 7, 7) with t = max(l), already padded
        frames = self.vis_dropout(feat_subgoal['frames'])
        # (B, t, args.dframe)
        vis_feat = self.vis_enc_step(frames)

        # Make object representation
        if self.object_repr == 'instance':
            # (B, t, dhid), (B, t, dhid)
            obj_visible = self.max_pool_object_features(
                feat_subgoal['object_token_id'], feat_subgoal['object_visibility'], 
                receptacle_indices=feat_subgoal['receptacle_token_id'], object_distances=feat_subgoal['object_distance'])
            obj_state_change = self.max_pool_object_features(
                feat_subgoal['object_token_id'], feat_subgoal['object_state_change'], 
                receptacle_indices=feat_subgoal['receptacle_token_id'], object_distances=feat_subgoal['object_distance'])            
        else: # 'type'
            # (B, t, dhid)
            obj_visible = self.max_pool_object_features(feat_subgoal['object_token_id'], feat_subgoal['object_visibility'])
            obj_state_change = self.max_pool_object_features(feat_subgoal['object_token_id'], feat_subgoal['object_state_change'])

        # ( B, t, args.demb+args.dframe+args.dhid+args.dhid)
        inp_seq = torch.cat([emb_act, vis_feat, obj_visible, obj_state_change], dim=2)
        packed_input = pack_padded_sequence(inp_seq, seq_lengths, batch_first=True, enforce_sorted=False)

        # Encode entire subgoal sequence
        # packed sequence object, (h_n, c_n) tuple
        enc_act, curr_subgoal_states = self.encoder(input=packed_input, hx=last_subgoal_hx)
        # (B, t, args.dhid*2)
        enc_act, _ = pad_packed_sequence(enc_act, batch_first=True)
        self.act_dropout(enc_act)

        # Apply learned self-attention
        # (B, args.dhid*2)
        cont_act = self.enc_att(enc_act)

        # (B, args.dhid*2), (B, t, args.dhid*2), (h_n, c_n)
        return cont_act, enc_act, curr_subgoal_states         


class MaskDecoder(nn.Module):
    '''
    mask decoder
    '''

    def __init__(self, dhid, pframe=300, hshape=(64,7,7)):
        super(MaskDecoder, self).__init__()
        self.dhid = dhid
        self.hshape = hshape
        self.pframe = pframe

        self.d1 = nn.Linear(self.dhid, hshape[0]*hshape[1]*hshape[2])
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn1 = nn.BatchNorm2d(16)
        self.dconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.dconv2 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.dconv1 = nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.d1(x))
        x = x.view(-1, *self.hshape)

        x = self.upsample(x)
        x = self.dconv3(x)
        x = F.relu(self.bn2(x))

        x = self.upsample(x)
        x = self.dconv2(x)
        x = F.relu(self.bn1(x))

        x = self.dconv1(x)
        x = F.interpolate(x, size=(self.pframe, self.pframe), mode='bilinear')

        return x


class ConvFrameMaskDecoder(nn.Module):
    '''
    action decoder
    '''

    def __init__(self, emb, dframe, dhid, pframe=300,
                 attn_dropout=0., hstate_dropout=0., actor_dropout=0., input_dropout=0.,
                 teacher_forcing=False):
        super().__init__()
        demb = emb.weight.size(1)

        self.emb = emb
        self.pframe = pframe
        self.dhid = dhid
        self.vis_encoder = ResnetVisualEncoder(dframe=dframe)
        self.cell = nn.LSTMCell(dhid+dframe+demb, dhid)
        self.attn = DotAttn()
        self.input_dropout = nn.Dropout(input_dropout)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.hstate_dropout = nn.Dropout(hstate_dropout)
        self.actor_dropout = nn.Dropout(actor_dropout)
        # a learned initial action embedding to speed up learning
        self.go = nn.Parameter(torch.Tensor(demb))
        self.actor = nn.Linear(dhid+dhid+dframe+demb, demb)
        self.mask_dec = MaskDecoder(dhid=dhid+dhid+dframe+demb, pframe=self.pframe)
        self.teacher_forcing = teacher_forcing
        self.h_tm1_fc = nn.Linear(dhid, dhid)

        nn.init.uniform_(self.go, -0.1, 0.1)

    def step(self, enc, frame, e_t, state_tm1):
        # previous decoder hidden state
        h_tm1 = state_tm1[0]

        # encode vision and lang feat
        vis_feat_t = self.vis_encoder(frame)
        lang_feat_t = enc # language is encoded once at the start

        # attend over language
        weighted_lang_t, lang_attn_t = self.attn(self.attn_dropout(lang_feat_t), self.h_tm1_fc(h_tm1))

        # concat visual feats, weight lang, and previous action embedding
        inp_t = torch.cat([vis_feat_t, weighted_lang_t, e_t], dim=1)
        inp_t = self.input_dropout(inp_t)

        # update hidden state
        state_t = self.cell(inp_t, state_tm1)
        state_t = [self.hstate_dropout(x) for x in state_t]
        h_t = state_t[0]

        # decode action and mask
        # (B, dhid+ dhid+dframe+demb)
        cont_t = torch.cat([h_t, inp_t], dim=1)
        action_emb_t = self.actor(self.actor_dropout(cont_t))
        action_t = action_emb_t.mm(self.emb.weight.t())
        mask_t = self.mask_dec(cont_t)

        return action_t, mask_t, state_t, lang_attn_t

    def forward(self, enc, frames, gold=None, max_decode=150, state_0=None):
        max_t = gold.size(1) if self.training else min(max_decode, frames.shape[1])
        batch = enc.size(0)
        e_t = self.go.repeat(batch, 1)
        state_t = state_0

        actions = []
        masks = []
        attn_scores = []
        for t in range(max_t):
            action_t, mask_t, state_t, attn_score_t = self.step(enc, frames[:, t], e_t, state_t)
            masks.append(mask_t)
            actions.append(action_t)
            attn_scores.append(attn_score_t)
            if self.teacher_forcing and self.training:
                w_t = gold[:, t]
            else:
                w_t = action_t.max(1)[1]
            e_t = self.emb(w_t)

        results = {
            'out_action_low': torch.stack(actions, dim=1),
            'out_action_low_mask': torch.stack(masks, dim=1),
            'out_attn_scores': torch.stack(attn_scores, dim=1),
            'state_t': state_t
        }
        return results


class ConvFrameMaskDecoderProgressMonitor(nn.Module):
    '''
    action decoder with subgoal and progress monitoring
    '''

    def __init__(self, emb, dframe, dhid, pframe=300,
                 attn_dropout=0., hstate_dropout=0., actor_dropout=0., input_dropout=0.,
                 teacher_forcing=False):
        super().__init__()
        demb = emb.weight.size(1)

        self.emb = emb
        self.pframe = pframe
        self.dhid = dhid
        self.vis_encoder = ResnetVisualEncoder(dframe=dframe)
        self.cell = nn.LSTMCell(dhid+dframe+demb, dhid)
        self.attn = DotAttn()
        self.input_dropout = nn.Dropout(input_dropout)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.hstate_dropout = nn.Dropout(hstate_dropout)
        self.actor_dropout = nn.Dropout(actor_dropout)
        self.go = nn.Parameter(torch.Tensor(demb))
        self.actor = nn.Linear(dhid+dhid+dframe+demb, demb)
        self.mask_dec = MaskDecoder(dhid=dhid+dhid+dframe+demb, pframe=self.pframe)
        self.teacher_forcing = teacher_forcing
        self.h_tm1_fc = nn.Linear(dhid, dhid)

        self.subgoal = nn.Linear(dhid+dhid+dframe+demb, 1)
        self.progress = nn.Linear(dhid+dhid+dframe+demb, 1)

        nn.init.uniform_(self.go, -0.1, 0.1)

    def step(self, enc, frame, e_t, state_tm1):
        # previous decoder hidden state
        h_tm1 = state_tm1[0]

        # encode vision and lang feat
        vis_feat_t = self.vis_encoder(frame)
        lang_feat_t = enc # language is encoded once at the start

        # attend over language
        weighted_lang_t, lang_attn_t = self.attn(self.attn_dropout(lang_feat_t), self.h_tm1_fc(h_tm1))

        # concat visual feats, weight lang, and previous action embedding
        inp_t = torch.cat([vis_feat_t, weighted_lang_t, e_t], dim=1)
        inp_t = self.input_dropout(inp_t)

        # update hidden state
        state_t = self.cell(inp_t, state_tm1)
        state_t = [self.hstate_dropout(x) for x in state_t]
        h_t, c_t = state_t[0], state_t[1]

        # decode action and mask
        cont_t = torch.cat([h_t, inp_t], dim=1)
        action_emb_t = self.actor(self.actor_dropout(cont_t))
        action_t = action_emb_t.mm(self.emb.weight.t())
        mask_t = self.mask_dec(cont_t)

        # predict subgoals completed and task progress
        subgoal_t = F.sigmoid(self.subgoal(cont_t))
        progress_t = F.sigmoid(self.progress(cont_t))

        return action_t, mask_t, state_t, lang_attn_t, subgoal_t, progress_t

    def forward(self, enc, frames, gold=None, max_decode=150, state_0=None):
        max_t = gold.size(1) if self.training else min(max_decode, frames.shape[1])
        batch = enc.size(0)
        e_t = self.go.repeat(batch, 1)
        state_t = state_0

        actions = []
        masks = []
        attn_scores = []
        subgoals = []
        progresses = []
        for t in range(max_t):
            action_t, mask_t, state_t, attn_score_t, subgoal_t, progress_t = self.step(enc, frames[:, t], e_t, state_t)
            masks.append(mask_t)
            actions.append(action_t)
            attn_scores.append(attn_score_t)
            subgoals.append(subgoal_t)
            progresses.append(progress_t)

            # find next emb
            if self.teacher_forcing and self.training:
                w_t = gold[:, t]
            else:
                w_t = action_t.max(1)[1]
            e_t = self.emb(w_t)

        results = {
            'out_action_low': torch.stack(actions, dim=1),
            'out_action_low_mask': torch.stack(masks, dim=1),
            'out_attn_scores': torch.stack(attn_scores, dim=1),
            'out_subgoal': torch.stack(subgoals, dim=1),
            'out_progress': torch.stack(progresses, dim=1),
            'state_t': state_t
        }
        return results


class LanguageDecoder(nn.Module):
    '''
    Language (instr / goal/ both) decoder
    '''

    def __init__(self, emb, dhid, attn_dropout=0., hstate_dropout=0.,
                 word_dropout=0., input_dropout=0., train_teacher_forcing=False, train_student_forcing_prob=0.1, 
                 obj_emb=None, aux_loss_over_object_states=False, object_repr=None, instance_fc=None):
        
        super().__init__()
        # args.demb
        demb = emb.weight.size(1)

        # embedding module for words
        self.emb = emb
        # embedding module for objects
        self.obj_emb = obj_emb
        # hidden layer size
        # 2*args.dhids
        self.dhid = dhid
        # LSTM cell, unroll one time step per call
        self.cell = nn.LSTMCell(dhid+demb, dhid)
        # attn to encoder output
        self.attn = DotAttn()
        # dropout concat input to LSTM cell
        self.input_dropout = nn.Dropout(input_dropout)
        # dropout on encoder output, before attn is applied
        self.attn_dropout = nn.Dropout(attn_dropout)
        # dropout on hidden state passed between steps
        self.hstate_dropout = nn.Dropout(hstate_dropout)
        # dropout on word fc
        self.word_dropout = nn.Dropout(word_dropout)
        # a learned initial word embedding to speed up learning
        self.go = nn.Parameter(torch.Tensor(demb))
        # word fc per time step
        # (1024 + 1024 + 100, 100)
        self.word = nn.Linear(dhid+dhid+demb, demb)
        self.train_teacher_forcing = train_teacher_forcing
        self.train_student_forcing_prob = train_student_forcing_prob
        self.h_tm1_fc = nn.Linear(dhid, dhid)

        # Aux Loss
        self.aux_loss_over_object_states = aux_loss_over_object_states
        if self.aux_loss_over_object_states:
            self.object_repr = object_repr
            self.h_enc_fc = nn.Linear(1, 2)
            self.instance_fc = instance_fc

        nn.init.uniform_(self.go, -0.1, 0.1)

    def step(self, enc, e_t, state_tm1):
        '''
        enc: (B, T, args.dhid). LSTM encoder per action output
        e_t: (B, demb) learned <go> embedding.
        '''

        # previous decoder hidden state
        # (B, 2*args.dhid) for bidirectional
        h_tm1 = state_tm1[0]

        # encode action feat
        # (B, t, 2*args.dhid) for bidirectional
        act_feat_t = enc # actions are encoded once at the start

        # attend over actions
        # inputs (B, t, 2*args.dhid), (B, 2*args.dhid)
        # output (B, 2*args.dhid), (B, t, 1)
        weighted_act_t, act_attn_t = self.attn(self.attn_dropout(act_feat_t), self.h_tm1_fc(h_tm1))

        # concat weight act, and previous word embedding
        # (B, 2*args.dhid + demb)
        inp_t = torch.cat([weighted_act_t, e_t], dim=1)
        inp_t = self.input_dropout(inp_t)

        # update hidden state
        # (B, 2*args.dhid, ), (B, 2*args.dhid, )
        state_t = self.cell(inp_t, state_tm1)
        state_t = [self.hstate_dropout(x) for x in state_t]
        h_t, c_t = state_t[0], state_t[1]

        # decode next word
        # (B, 2*args.dhid + 2*args.dhid + demb)
        cont_t = torch.cat([h_t, inp_t], dim=1)
        # (B, demb)
        word_emb_t = self.word(self.word_dropout(cont_t))
        # (B, demb).mm(demb, vocab size) = (B, vocab size)
        word_t = word_emb_t.mm(self.emb.weight.t())
 
        return word_t, state_t, act_attn_t

    def make_instance_embeddings(self, object_indices, receptacle_indices, object_distances):
        '''
        object_indices: (B, max_num_objects in batch)
        receptacle_indices: (B, max_num_objects in batch)
        object_distance: (B, max_num_objects in batch)
        '''
        # (B, max_num_objects in batch, obj demb)
        obj_embeddings = self.obj_emb(object_indices)      
        # (B, max_num_objects in batch, obj demb)
        recep_embeddings = self.obj_emb(receptacle_indices)
        # (B, max_num_objects in batch, obj demb + obj demb + 1)
        cat_embeddings = torch.cat([obj_embeddings, recep_embeddings, object_distances.unsqueeze(-1)], dim=len(obj_embeddings.shape) - 1)
        # (B, max_num_objects in batch, dhid)
        return self.instance_fc(cat_embeddings)

    def forward(self, enc, feat_subgoal, max_decode=50, state_0=None,  valid_object_indices=None,
                validate_teacher_forcing=False, validate_sample_output=False):
        '''
        enc :                     (B, T, args.dhid). LSTM encoder per action output
        gold:                     (B, T). padded_sequence of word index tokens.
        max_decode:               integer. maximum timesteps - length of language instruction.
        state_0:                  tuple (cont_act, torch.zeros.like(cont_act)).
        valid_object_indices:     (B, max_num_objects of batch)
        validate_teacher_forcing: boolean. Whether to use ground-truth to score loss and perplexity. 
                                  Student-forcing is used otherwise.
        validate_sample_output:   boolean. With student-forcing, whether to decode by sampling (1) or argmax (0).
        '''

        # Aux Loss---------------------------------------------------
        # (B, 2*args.dhid)
        h_enc_tm1 = state_0[0]

        # Simple Aux Loss prediction using dot products
        obj_visibilty_scores, obj_state_change_scores = None, None
        if self.aux_loss_over_object_states:

            # Max pool hidden state 
            # (B, args.dhid)
            h_enc_max_pooled = torch.max(h_enc_tm1.reshape(h_enc_tm1.shape[0], 2, -1), dim=1)[0]
            assert h_enc_max_pooled.shape == (h_enc_tm1.shape[0], h_enc_tm1.shape[1]/2)

            # Expand by linear layer
            # (B, args.dhid, 1) -> (B, args.dhid, 2)
            h_enc_extended = self.h_enc_fc(h_enc_max_pooled.unsqueeze(-1))
            # (B, 2, args.dhid)
            h_enc_extended = h_enc_extended.transpose(1, 2)

            if self.object_repr == 'instance':
                # Taken at beginning of subgoal with [:,0,:]
                # (B, max_num_objects, args.dhid)
                obj_m = self.make_instance_embeddings(
                    object_indices=feat_subgoal['object_token_id'][:, 0, :],  # (B, max_num_objects)
                    receptacle_indices=feat_subgoal['receptacle_token_id'][:, 0, :],  # (B, max_num_objects)
                    object_distances=feat_subgoal['object_distance'][:, 0, :])  # (B, max_num_objects)
                # (B, args.dhid, max_num_objects)
                obj_m = obj_m.transpose(1, 2)
            else: # 'type'
                # Taken at beginning of subgoal
                # (B, max_num_objects)
                object_indices = feat_subgoal['object_token_id'][:, 0, :]
                # (B, args.dhid, max_num_objects)
                obj_m = self.obj_emb(object_indices).transpose(1, 2)

            # obj_m = self.obj_emb.weight.t().unsqueeze(0).repeat(h_enc_tm1.shape[0],1,1)
            # (B, 2, args.dhid) bmm (B, args.dhid, max_num_objects) = (B, 2, max_num_objects)
            aux_scores = h_enc_extended.bmm(obj_m)
            # (B, max_num_objects), (B, max_num_objects)
            obj_visibilty_scores, obj_state_change_scores = aux_scores[:,0,:], aux_scores[:,1,:]                

        # deal with valid object indices in compute_loss

        # Language Model----------------------------------------------
        gold = feat_subgoal['lang_instr']
        max_t = gold.size(1) if (self.training or validate_teacher_forcing) else max_decode
        batch = enc.size(0)
        # go is a learned embedding
        e_t = self.go.repeat(batch, 1)
        state_t = state_0

        words = []
        attn_scores = []

        for t in range(max_t):

            word_t, state_t, attn_score_t  = self.step(enc, e_t, state_t)
            words.append(word_t)
            attn_scores.append(attn_score_t)

            # next word choice
            if self.training:
                if self.train_teacher_forcing:
                    w_t = gold[:, t]
                else:
                    use_student = np.random.binomial(1, self.train_student_forcing_prob)
                    if use_student:
                        w_t = word_t.max(1)[1]
                    else:
                        w_t = gold[:, t]
            else:
                if validate_teacher_forcing:
                    w_t = gold[:, t]
                else:
                    if validate_sample_output:
                        w_t = torch.multinomial(torch.exp(word_t), 1).squeeze(-1)
                    else:
                        # argmax
                        w_t = word_t.max(1)[1]

            # next word embedding
            e_t = self.emb(w_t)
 
        results = {
            # shape (B, T , Vocab size)
            'out_lang_instr': torch.stack(words, dim=1),
            'out_attn_scores': torch.stack(attn_scores, dim=1),
            'state_t': state_t,
            'out_obj_vis_score': obj_visibilty_scores,
            'out_state_change_score': obj_state_change_scores,
            'valid_object_indices': valid_object_indices
        }

        return results, state_t