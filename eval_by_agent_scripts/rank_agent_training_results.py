import json
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser


if __name__ == '__main__':
    # parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    # settings
    parser.add_argument('--exp_dir',type=str)
    parser.add_argument('--mod_name',type=str)

    args = parser.parse_args()
	
    dir_p = os.path.join(args.exp_dir, args.mod_name)
	
    # dir_p = /root/home/hoyeung/blob_alfred_data/exp_all/ 

    valid_seen_f1s = []
    valid_seen_losses = []
    valid_seen_exact_match = []
    valid_unseen_f1s = []
    valid_unseen_losses = []
    valid_unseen_exact_match = []

    for e in range(30):
        fname = 'stats_epoch_' + str(e) + '.json'
        try:
            f_p = os.path.join(dir_p, fname)
            with open(f_p, 'r') as f:
                stats = json.load(f)
            valid_seen_f1s.append(stats['valid_seen']['action_low_f1'])
            valid_seen_losses.append(stats['valid_seen']['total_loss'])
            valid_seen_exact_match.append(stats['valid_seen']['action_low_em'])
            valid_unseen_f1s.append(stats['valid_unseen']['action_low_f1'])
            valid_unseen_losses.append(stats['valid_unseen']['total_loss'])
            valid_unseen_exact_match.append(stats['valid_seen']['action_low_em'])
            
        except:
            pass

    sort_f1 = lambda ls : sorted([(e,v) for e,v in enumerate(ls)], key=lambda x:x[1], reverse=True)
    sort_em = lambda ls : sorted([(e,v) for e,v in enumerate(ls)], key=lambda x:x[1], reverse=True)
    sort_loss = lambda ls : sorted([(e,v) for e,v in enumerate(ls)], key=lambda x:x[1], reverse=False)

    print('----------------------------------')
    print('valid seen F1 ranking:')
    print(sort_f1(valid_seen_f1s))    
    print('----------------------------------')
    print('valid seen exact match ranking:')
    print(sort_em(valid_seen_exact_match))
    print('----------------------------------')
    print('valid seen loss ranking:')
    print(sort_loss(valid_seen_losses))
    print('----------------------------------')
    print('valid unseen F1 ranking:')
    print(sort_f1(valid_unseen_f1s))
    print('----------------------------------')
    print('valid unseen exact match ranking:')
    print(sort_em(valid_unseen_exact_match))
    print('----------------------------------')
    print('valid unseen loss ranking:')
    print(sort_loss(valid_unseen_losses))