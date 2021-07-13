'''Evaluate Linguistics metrics for a language model given its predictions on valid seen and valid unseen data'''

import os
import json
import pprint
import numpy as np

from collections import defaultdict
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from nltk.translate.bleu_score import sentence_bleu

import spacy
nlp = spacy.load("en_core_web_sm")

#####################################################################################
cats = (
    'GotoLocation', 'SliceObject', 'PickupObject', 'PutObject', 
    'ToggleObject', 'CleanObject', 'HeatObject', 'CoolObject'
)

# Map object type to variations
type_to_variations = {
    ' alarmclock ' : [' alarm clock ', ' clock ', ' alarm '],
    ' armchair ' : [' arm chair '],
    ' couch ' : ['sofa'],
    ' baseballbat ': [' baseball bat ', ' bat '],
    ' basketball ' : [' basket ball '],
    ' bathtub ' : [' bath tub ', ' bathtubbasin ', ' bathtub basin '],
    ' cabinet ': [' cupboard '],
    ' cellphone ': [' cell phone ', ' phone ', ' mobilephone ', ' mobile phone '],
    ' coffeemachine ': [' coffee machine ', ' coffeemaker ', ' coffee maker ', ' coffee maker machine '],
    ' countertop ': [' counter ', ' counter top ', ' island ', ' kitchen island '],
    ' creditcard ': [' credit card ', ' card '],
    ' cup ': [' mug '],
    ' curtains ': [' curtain '],
    ' table ': [' desk ', ' study desk ', ' studydesk ',' work desk ', ' workdesk '],
    ' desk lamp ': [' desklamp ', ' desk light ', ' desklight '],
    ' dishsponge ': [' dish sponge ', ' sponge '],
    ' floorlamp ': [' floor lamp ', ' lamp '],
    ' footstool ': [' foot stool ', ' stool '],
    ' fridge ': [' refrigerator '],
    ' garbagecan ': [' garbage can ', ' garbage bin ', ' trashcan ', ' trash can ', ' trash bin '],
    ' bottle ': [' glass bottle ', ' bottle ', ' glassbottle '],
    ' handtowel ': [' hand towel ', ' towel '],
    ' towelholder ': [' handtowel holder ', ' towel holder ', ' towelholder ', ' hand towel holder ', ' towelrack ', ' towel rack '],
    ' houseplant ': [' house plant ', ' plant '],
    ' knife ': [' butterknife ', ' butter knife '],
    ' laptop ': [' computer ', ' lap top '],
    ' laundryhamper ': [' laundry hamper ', ' hamper ', ' laundry basket ', ' laundrybasket '],
    ' laundryhamperlid ': [' laundry hamper lid ', ' laundry basket lid ', ' hamper lid ', ' basket lid '],
    ' lightswitch ': [' light switch '],
    ' mug ': [' cup '],
    ' papertowel ': [' paper towel '],
    ' papertowelroll ': [' papertowel roll ', ' paper towel roll '],
    ' peppershaker ': [' pepper shaker '],
    ' remotecontrol ': [' remote control ', ' remote '],
    ' saltshaker ': [' salt shaker '],
    ' brush ': [' scrub brush ', ' scrubbrush '],
    ' showerdoor ': [' shower door '],
    ' showerglass ': [' shower glass '],
    ' sinkbasin ': [' sink basin ', ' basin ', ' sink '],
    ' soapbar ': [' soap bar ', ' soap '],
    ' soapbottle ': [' soap bottle '],
    ' spraybottle ': [' spray bottle '],
    ' stoveburner ': [' stove burner ', ' stove ', ' burner '],
    ' stoveknob ': [' stove knob ', ' stove switch ', ' knob ', ' switch '],
    ' diningtable ': [' dining table '],
    ' coffeetable ': [' coffee table '],
    ' sidetable ': [' side table '],
    ' teddybear ': [' teddy bear ', ' bear '],
    ' tennisracket ': [' tennis racket ', ' racket '],
    ' tissuebox ': [' tissue box '],
    ' toiletpaper ': [' toilet paper '],
    ' toiletpaperhanger ': [' toilet paper hanger ', ' toiletpaper hanger '],
    ' toiletpaperroll ': [' toilet paper roll ', ' toiletpaper roll '],
    ' tvstand ': [' tv stand ', ' television stand ', ' televisionstand '],
    ' tv ': [' television '],
    ' wateringcan ': [' watering can '],
    ' winebottle ': [' wine bottle '],
}

# Map object variations to type
variation_to_type = {}
all_phrases_set = set()
for k,vals in type_to_variations.items():
    all_phrases_set.add(k.strip())
    for v in vals:
        variation_to_type[v] = k

# list of variations to scan each sentence
sorted_phrases = sorted(variation_to_type.keys(), key=lambda x: len(x), reverse=True)

#####################################################################################

def get_traj_data(root):
    traj_path = os.path.join(root, 'pp', 'ann_0.json')
    with open(traj_path, 'r') as f:
        return json.load(f)

def get_subgoals(key, split, dat):
    root = dat[key]['root']
    traj_data = get_traj_data(root)
    gold_num_subgoals = len(traj_data['num']['action_high'])-1
    return gold_num_subgoals

def get_ref_instrs(root):
    ref_instrs = []
    for i in range(3):
        path = os.path.join(root, 'pp', 'ann_%d.json' % i)
        with open(path, 'r') as f:
            ex = json.load(f)
            ref_instrs.append([[word.strip() for word in subgoal if word.strip()!='.'] for subgoal in ex['ann']['instr']])
    return ref_instrs

def tokensize_and_remove_fullstop(sent):
    words = sent.split(' ')
    if '.' in words:
        words.remove('.')
    return words

def extract_nouns(tokens):
    # if these words are picked out as nouns because the annotation was poorly written, 
    # remove them from final set
    remove_nouns = {'left', 'right', 'top', 'down', 'front', 'back', 
                    'step', 'steps', 'remove', 'side', 'turn', 'middle', 
                    'face', 'head', 'end', 'walk', 'move'}
    
    sent = ' ' + ' '.join(tokens) + ' '

    # Map linguistic variations of an object back to one noun
    cleaned_sent = sent
    for phrase in sorted_phrases:
        # ToDo -- referring to global variation_to_type
        cleaned_sent = cleaned_sent.replace(phrase, variation_to_type[phrase])
    
    doc = nlp(cleaned_sent)
    
    # pick out the nouns
    nouns = [str(token) for token in doc if token.pos_ == 'NOUN' and not (str(token) in remove_nouns)]
    
    # if no nouns were picked, sometimes it's because the sentence was poorly written by annotators
    # try one last trick to check if there's overlap with env object set
    if len(nouns) == 0:
        resplit_tokens = cleaned_sent.strip().split(' ')
        nouns = (set(resplit_tokens) & all_phrases_set)
        
    return set(nouns)

def compute_precision(ann_noun_set, pred_noun_set):
    ct = 0
    matched = len(pred_noun_set) - len(pred_noun_set - ann_noun_set)
    return matched*1.0/len(pred_noun_set) if len(pred_noun_set) != 0 else 0

def compute_recall(ann_noun_set, pred_noun_set):
    ct = 0
    matched = len(ann_noun_set) - len(ann_noun_set - pred_noun_set)
    return matched*1.0/len(ann_noun_set) if len(ann_noun_set) != 0 else 0

def compute_f1(anns, main_pred):
    '''
    anns : list of 3 lists of tokens
    main_pred : list of tokens
    '''
    precisions = []
    recalls = []
    f1s = []
    predicted_nouns = extract_nouns(main_pred)
    for ann in anns:
        ann_nouns = extract_nouns(ann)
        precision = compute_precision(ann_nouns, predicted_nouns)
        
        recall = compute_recall(ann_nouns, predicted_nouns)
            
        precisions.append(precision)
        recalls.append(recall)
        
        if precision+recall != 0:
            f1 = 2*(precision*recall)/(precision+recall)
        else:
            f1 = 0
            
        f1s.append(f1)
    return sum(precisions)/len(precisions), sum(recalls)/len(recalls), sum(f1s)/len(f1s)

def compute_f1(anns, main_pred):
    '''
    anns : list of 3 lists of tokens
    main_pred : list of tokens
    '''
    precisions = []
    recalls = []
    f1s = []
    predicted_nouns = extract_nouns(main_pred)
    for ann in anns:
        ann_nouns = extract_nouns(ann)
        precision = compute_precision(ann_nouns, predicted_nouns)
        
        recall = compute_recall(ann_nouns, predicted_nouns)
            
        precisions.append(precision)
        recalls.append(recall)
        
        if precision+recall != 0:
            f1 = 2*(precision*recall)/(precision+recall)
        else:
            f1 = 0
            
        f1s.append(f1)
    return sum(precisions)/len(precisions), sum(recalls)/len(recalls), sum(f1s)/len(f1s)

def eval_results(split_str, split_keys, main_dat, human_ann, lmtag='Model'):
    '''Compute main metrics: BLEU, Object Capture Test F1, Precision, Recall'''
    
    subgoals = ['All'] + cats
    
    score_keys = {subgoal:[] for subgoal in subgoals}
    BLEU_all = {subgoal:{lmtag:[], 'Annotation':[]} for subgoal in subgoals}
    BLEU_mean = {subgoal:{lmtag:[], 'Annotation':None} for subgoal in subgoals}
    
    RECALL_all = {subgoal:{lmtag:[],  'Annotation':[]} for subgoal in subgoals}
    PRECISION_all = {subgoal:{lmtag:[], 'Annotation':[]} for subgoal in subgoals}
    F1_all = {subgoal:{lmtag:[], 'Annotation':[]} for subgoal in subgoals}
    RECALL_mean = {subgoal:{lmtag:None} for subgoal in subgoals}
    PRECISION_mean = {subgoal:{lmtag:None} for subgoal in subgoals}
    F1_mean = {subgoal:{lmtag:None} for subgoal in subgoals}

    sent_lens = {subgoal:{lmtag:[], 'Annotation':[]} for subgoal in subgoals}
    word_counts = {subgoal:{lmtag:defaultdict(int), 'Annotation':defaultdict(int)} for subgoal in subgoals} 
    
    for i, key in enumerate(split_keys):
        gold_num_subgoals = get_subgoals(key, split_str, main_dat)
        gold_subgoal_names = main_dat[key]['action_high'][:-1]
        
        # get LM preds
        main_pred = main_dat[key]['p_lang_instr']
        main_pred = {subgoal_idx:tokensize_and_remove_fullstop(main_pred[str(subgoal_idx)]) for subgoal_idx in range(gold_num_subgoals)}
        # get human
        anns = {i:[] for i in range(gold_num_subgoals)}
        for j, ann in enumerate(human_ann[key[:-2]]):
            success = True
            for i in range(gold_num_subgoals):
                try:
                    anns[i].append(ann[i])
                except:
                    success = False
                    break
            assert success
            
        # check if numbers of subgoals match
        assert len(main_pred) == len(anns) == gold_num_subgoals
        
        for subgoal_i in range(gold_num_subgoals):
            # --------------------------------------------------------------------
            # Compute BLEU 
            subgoal_name = gold_subgoal_names[subgoal_i]
            score_keys[subgoal_name].append((key, subgoal_i)) 
            
            main_bleu = sentence_bleu(anns[subgoal_i], main_pred[subgoal_i])
            #ann_bleu = compute_human_bleu(anns[subgoal_i])

            BLEU_all[subgoal_name][lmtag].append(main_bleu)
            BLEU_all['All'][lmtag].append(main_bleu)
            
            # --------------------------------------------------------------------
            sent_lens[subgoal_name][lmtag].append(len(main_pred[subgoal_i]))
            sent_lens[subgoal_name]['Annotation'].extend([len(a) for a in anns[subgoal_i]])
            
            for w in main_pred[subgoal_i]:
                word_counts[subgoal_name][lmtag][w] += 1
            
            for ann in anns[subgoal_i]:
                for w in ann:
                    word_counts[subgoal_name]['Annotation'][w] += 1
            # --------------------------------------------------------------------
            # compute F-1
            main_precision, main_recall, main_f1 = compute_f1(anns[subgoal_i], main_pred[subgoal_i])            
        
            RECALL_all[subgoal_name][lmtag].append(main_recall)
            PRECISION_all[subgoal_name][lmtag].append(main_precision) 
            F1_all[subgoal_name][lmtag].append(main_f1)

            RECALL_all['All'][lmtag].append(main_recall)
            PRECISION_all['All'][lmtag].append(main_precision) 
            F1_all['All'][lmtag].append(main_f1)
            
        # --------------------------------------------------------------------
        
    for subgoal_name in subgoals:
        BLEU_mean[subgoal_name][lmtag] = sum(BLEU_all[subgoal_name][lmtag])/len(BLEU_all[subgoal_name][lmtag])
        RECALL_mean[subgoal_name][lmtag] = sum(RECALL_all[subgoal_name][lmtag])/len(RECALL_all[subgoal_name][lmtag])
        PRECISION_mean[subgoal_name][lmtag] = sum(PRECISION_all[subgoal_name][lmtag])/len(PRECISION_all[subgoal_name][lmtag])
        F1_mean[subgoal_name][lmtag] = sum(F1_all[subgoal_name][lmtag])/len(F1_all[subgoal_name][lmtag])
                
    return {
        'BLEU_all':BLEU_all, 'RECALL_all':RECALL_all, 'PRECISION_all':PRECISION_all, 
        'F1_all':F1_all, 'sent_lens':sent_lens, 'word_counts':word_counts,
        'score_keys':score_keys, 'BLEU_mean':BLEU_mean, 'RECALL_mean':RECALL_mean,
        'PRECISION_mean':PRECISION_mean, 'F1_mean':F1_mean
        }

def load_human_annotations(dat):
    human_anno = {}
    keys = set()
    for key, dat in dat.items():
        root = dat['root'] # TODO fix root
        key = '_'.join(key.split('_')[:-1])
        if key not in keys:
            ref_instrs = get_ref_instrs(root)
            human_anno[key] = ref_instrs
            keys.add(key)

def main(parse_args):

    # read prediction data and keys
    with open(parse_args.valid_seen_predictions_path, 'r') as f:
        valid_seen = json.load(f)
        valid_seen_keys = list(valid_seen.keys())

    with open(parse_args.valid_unseen_predictions_path, 'r') as f:
        valid_unseen = json.load(f)
        valid_unseen_keys = list(valid_unseen.keys())

    # read human annotations
    human_anno_valid_seen = load_human_annotations(valid_seen)
    human_anno_valid_unseen = load_human_annotations(valid_unseen)

    # run evaluation
    metrics_valid_seen = \
        eval_results('valid_seen', valid_seen_keys, valid_seen, human_anno_valid_seen)

    metrics_valid_unseen = \
        eval_results('valid_unseen', valid_unseen_keys, valid_unseen, human_anno_valid_unseen)

    # save out the results
    with open(parse_args.results_path, 'w') as f:
        json.dump({'valid_seen':metrics_valid_seen, 'valid_unseen':metrics_valid_unseen}, f)
    print(f'Saved evaluation results to {parse_args.results_path}')

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument(
        '--valid_seen_predictions_path', type=str, required=True,
        help='path to prediction file generated during validation on valid seen set. \
            e.g. dir/valid_seen.debug_epoch_32.preds.json'
    )
    parser.add_argument(
        '--valid_unseen_predictions_path', type=str, required=True,
        help='path to prediction file generated during validation on valid unseen set. \
            e.g. dir/valid_unseen.debug_epoch_29.preds.json'
    )
    
    parse_args = parser.parse_args()
    parse_args.results_dir = '/'.join(parse_args.valid_unseen_predictions_path.split('/')[:-1])
    parse_args.results_path = os.path.join(parse_args.results_dir, 'linguistic_eval_results.json')

    main(parse_args)