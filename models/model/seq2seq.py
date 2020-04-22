import os
import random
import json
import time
import torch
import pprint
import collections
import numpy as np
from torch import nn
from collections import defaultdict
from tensorboardX import SummaryWriter
from tqdm import trange

class Module(nn.Module):

    def __init__(self, args, vocab):
        '''
        Base Seq2Seq agent with common train and val loops
        '''
        super().__init__()

        # sentinel tokens
        self.pad = 0
        self.seg = 1 

        # args and vocab
        self.args = args
        self.vocab = vocab

        # emb modules
        self.emb_word = nn.Embedding(len(vocab['word']), args.demb)
        self.emb_action_low = nn.Embedding(len(vocab['action_low']), args.demb)

        # end tokens TODO need to replace with stop token of language
        self.stop_token = self.vocab['word'].word2index("<<stop>>", train=False)
        # self.seg_token = self.vocab['action_low'].word2index("<<seg>>", train=False)  # obsolete?

        # set random seed (Note: this is not the seed used to initialize THOR object locations)
        random.seed(a=args.seed)

        # initialize summary writer for tensorboardX
        self.summary_writer = SummaryWriter(log_dir=args.dout)

    def run_train(self, splits, args=None, optimizer=None, start_epoch=0):
        '''
        training loop
        '''

        # time
        time_report = defaultdict(int)
        # time
        start_time = time.time()

        # args
        args = args or self.args

        # splits
        train = splits['train']
        train_sanity = splits['train_sanity']
        # ann_0, ann_1 and ann_2 have the same action sequence, only ann_0 is needed for validation
        valid_seen = [t for t in splits['valid_seen'] if t['repeat_idx'] == 0]
        valid_unseen = [t for t in splits['valid_unseen'] if t['repeat_idx'] == 0]

        # debugging: chose a small fraction of the dataset
        if self.args.dataset_fraction > 0:
            small_train_size = int(self.args.dataset_fraction * 0.7)
            small_valid_size = int((self.args.dataset_fraction * 0.3) / 2)
            train = train[:small_train_size]
            train_sanity = train_sanity[:small_valid_size]
            valid_seen = valid_seen[:small_valid_size]
            valid_unseen = valid_unseen[:small_valid_size]

        # debugging: use to check if training loop works without waiting for full epoch
        if self.args.fast_epoch:
            train = train[:16]
            train_sanity = train_sanity[:16]
            valid_seen = valid_seen[:16]
            valid_unseen = valid_unseen[:16]

        # dump config
        fconfig = os.path.join(args.dout, 'config.json')
        with open(fconfig, 'wt') as f:
            json.dump(vars(args), f, indent=2)

        # optimizer
        optimizer = optimizer or torch.optim.Adam(self.parameters(), lr=args.lr)

        # time
        time_report['setup_time'] += time.time() - start_time

        # display dout
        print("Saving to: %s" % self.args.dout)
        best_metric = {'train': -1e10, 'train_sanity': -1e10, 'valid_seen': -1e10, 'valid_unseen': -1e10}
        train_iter, train_sanity_iter, valid_seen_iter, valid_unseen_iter = 0, 0, 0, 0
        for epoch in trange(start_epoch, args.epoch, desc='epoch'):
            # time
            epoch_start_time = time.time()
            m_train = collections.defaultdict(list)
            self.train()
            self.adjust_lr(optimizer, args.lr, epoch, decay_epoch=args.decay_epoch)
            p_train = {}
            total_train_loss = list()
            random.shuffle(train) # shuffle every epoch
            # time
            start_time = time.time()
            for batch, feat in self.iterate(train, args.batch):
                out = self.forward(feat)
                preds = self.extract_preds(out, batch, feat)
                p_train.update(preds)
                loss, perplexity = self.compute_loss(out, batch, feat)
                for k, v in loss.items():
                    ln = 'batch_loss_' + k
                    m_train[ln].append(v.item())
                    self.summary_writer.add_scalar('train/' + ln, v.item(), train_iter)
                m_train['perplexity'].append(perplexity.item())
                self.summary_writer.add_scalar('train/batch_perplexity', perplexity.item(), train_iter)

                # optimizer backward pass
                optimizer.zero_grad()
                sum_loss = sum(loss.values())
                sum_loss.backward()
                optimizer.step()

                self.summary_writer.add_scalar('train/batch_loss', sum_loss, train_iter)
                sum_loss = sum_loss.detach().cpu()
                total_train_loss.append(float(sum_loss))
                train_iter += self.args.batch
            # time
            time_report['forward_batch_train'] += time.time() - start_time

            # compute metrics for train, teacher-forcing
            # time
            start_time = time.time()
            m_train = {k: sum(v) / len(v) for k, v in m_train.items()}
            self.summary_writer.add_scalar('train/epoch_perplexity', m_train['perplexity'], train_iter)
            m_train['total_loss'] = sum(total_train_loss) / len(total_train_loss)
            self.summary_writer.add_scalar('train/epoch_total_loss', m_train['total_loss'], train_iter)
            # time
            time_report['compute_metrics_train'] += time.time() - start_time

            # time
            start_time = time.time()
            #-------------------------------------------------------
            # compute metrics for train_sanity, teacher forcing
            _, _, total_train_sanity_loss, m_train_sanity_teacher = self.run_pred(train_sanity, args=args, name='train_sanity', iter=train_sanity_iter, validate_teacher_forcing=True)
            m_train_sanity_teacher['total_loss'] = float(total_train_sanity_loss)
            self.summary_writer.add_scalar('train_sanity/epoch_perplexity_teacher_forcing', m_train_sanity_teacher['perplexity'], epoch)
            self.summary_writer.add_scalar('train_sanity/epoch_total_loss_teacher_forcing', m_train_sanity_teacher['total_loss'], epoch)

            # compute metrics for valid_seen, teacher forcing
            _, _, total_valid_seen_loss, m_valid_seen_teacher = self.run_pred(valid_seen, args=args, name='valid_seen', iter=valid_seen_iter, validate_teacher_forcing=True)
            m_valid_seen_teacher['total_loss'] = float(total_valid_seen_loss)
            self.summary_writer.add_scalar('valid_seen/epoch_perplexity_teacher_forcing', m_valid_seen_teacher['perplexity'], epoch)
            self.summary_writer.add_scalar('valid_seen/epoch_total_loss_teacher_forcing', m_valid_seen_teacher['total_loss'], epoch)

            # compute metrics for valid_unseen, teacher forcing
            _, _, total_valid_unseen_loss, m_valid_unseen_teacher = self.run_pred(valid_unseen, args=args, name='valid_unseen', iter=valid_unseen_iter, validate_teacher_forcing=True)
            m_valid_unseen_teacher['total_loss'] = float(total_valid_unseen_loss)
            self.summary_writer.add_scalar('valid_unseen/epoch_perplexity_teacher_forcing', m_valid_unseen_teacher['perplexity'], epoch)
            self.summary_writer.add_scalar('valid_unseen/epoch_total_loss_teacher_forcing', m_valid_unseen_teacher['total_loss'], epoch)

            #-------------------------------------------------------
            # compute metrics for train_sanity, student forcing, argmax
            p_train_sanity, train_sanity_iter, _, m_train_sanity_student = self.run_pred(train_sanity, args=args, name='train_sanity', iter=train_sanity_iter)
            m_train_sanity_student.update(self.compute_metric(p_train_sanity, train_sanity))
            self.summary_writer.add_scalar('train_sanity/epoch_BLEU_student_forcing', m_train_sanity_student['BLEU'], epoch)

            # compute metrics for valid_seen, student forcing, argmax
            p_valid_seen, valid_seen_iter, _, m_valid_seen_student = self.run_pred(valid_seen, args=args, name='valid_seen', iter=valid_seen_iter)
            m_valid_seen_student.update(self.compute_metric(p_valid_seen, valid_seen))
            self.summary_writer.add_scalar('valid_seen/epoch_BLEU_student_forcing', m_valid_seen_student['BLEU'], epoch)

            # compute metrics for valid_unseen, student forcing, argmax
            p_valid_unseen, valid_unseen_iter, _, m_valid_unseen_student = self.run_pred(valid_unseen, args=args, name='valid_unseen', iter=valid_unseen_iter)
            m_valid_unseen_student.update(self.compute_metric(p_valid_unseen, valid_unseen))
            self.summary_writer.add_scalar('valid_unseen/epoch_BLEU_student_forcing', m_valid_unseen_student['BLEU'], epoch)

            #-------------------------------------------------------
            m_train_sanity, m_valid_seen, m_valid_unseen = {}, {}, {}

            m_train_sanity['perplexity'], m_valid_seen['perplexity'], m_valid_unseen['perplexity'] = \
                m_train_sanity_teacher['perplexity'], m_valid_seen_teacher['perplexity'], m_valid_unseen_teacher['perplexity']
            m_train_sanity['total_loss'], m_valid_seen['total_loss'], m_valid_unseen['total_loss'] = \
                m_train_sanity_teacher['total_loss'], m_valid_seen_teacher['total_loss'], m_valid_unseen_teacher['total_loss']
            m_train_sanity['BLEU'], m_valid_seen['BLEU'], m_valid_unseen['BLEU'] = \
                m_train_sanity_student['BLEU'], m_valid_seen_student['BLEU'], m_valid_unseen_student['BLEU']

            stats = {'epoch': epoch, 'train_sanity': m_train_sanity, 'valid_seen': m_valid_seen, 'valid_unseen': m_valid_unseen}
            #-------------------------------------------------------
            # time
            time_report['compute_metrics_validation_sets'] += time.time() - start_time

            # new best valid_seen metric
            if m_valid_seen['BLEU'] > best_metric['valid_seen']:
                # time
                start_time = time.time()
                print('\nFound new best valid_seen!! Saving...')
                fsave = os.path.join(args.dout, 'best_seen.pth')
                torch.save({
                    'metric': stats,
                    'model': self.state_dict(),
                    'optim': optimizer.state_dict(),
                    'args': self.args,
                    'vocab': self.vocab,
                    'epoch': epoch,
                }, fsave)
                fbest = os.path.join(args.dout, 'best_seen.json')
                with open(fbest, 'wt') as f:
                    json.dump(stats, f, indent=2)
                best_metric['valid_seen'] = m_valid_seen['BLEU']
                # time
                time_report['torch_save_valid_seen'] += time.time() - start_time

            # make debugging output for valid_seen with student-forced predictions
            # time
            start_time = time.time()
            fpred = os.path.join(args.dout, 'valid_seen.debug_epoch_{}.preds.json'.format(epoch))
            with open(fpred, 'wt') as f:
                json.dump(self.make_debug(p_valid_seen, valid_seen), f, indent=2)
            # time
            time_report['make_debug_valid_seen'] += time.time() - start_time

            # new best valid_unseen metric
            if m_valid_unseen['BLEU'] > best_metric['valid_unseen']:
                # time
                start_time = time.time()
                print('Found new best valid_unseen!! Saving...')
                fsave = os.path.join(args.dout, 'best_unseen.pth')
                torch.save({
                    'metric': stats,
                    'model': self.state_dict(),
                    'optim': optimizer.state_dict(),
                    'args': self.args,
                    'vocab': self.vocab,
                    'epoch': epoch,
                }, fsave)
                fbest = os.path.join(args.dout, 'best_unseen.json')
                with open(fbest, 'wt') as f:
                    json.dump(stats, f, indent=2)
                best_metric['valid_unseen'] = m_valid_unseen['BLEU']
                # time
                time_report['torch_save_valid_unseen'] += time.time() - start_time

            # make debugging output for valid_seen with student-forced predictions
            # time
            start_time = time.time()
            fpred = os.path.join(args.dout, 'valid_unseen.debug_epoch_{}.preds.json'.format(epoch))
            with open(fpred, 'wt') as f:
                json.dump(self.make_debug(p_valid_unseen, valid_unseen), f, indent=2)
            # time
            time_report['make_debug_valid_unseen'] += time.time() - start_time

            # new best train_sanity metric
            if m_train_sanity['BLEU'] > best_metric['train_sanity']:
                # time
                start_time = time.time()
                print('Found new best train_sanity!! Saving...')
                fsave = os.path.join(args.dout, 'best_train_sanity.pth')
                torch.save({
                    'metric': stats,
                    'model': self.state_dict(),
                    'optim': optimizer.state_dict(),
                    'args': self.args,
                    'vocab': self.vocab,
                    'epoch': epoch,
                }, fsave)
                fbest = os.path.join(args.dout, 'best_train_sanity.json')
                with open(fbest, 'wt') as f:
                    json.dump(stats, f, indent=2)
                best_metric['train_sanity'] = m_train_sanity['BLEU']
                # time
                time_report['torch_save_train_sanity'] += time.time() - start_time

            # make debugging output for train_sanity with student-forced predictions
            # time
            start_time = time.time()
            fpred = os.path.join(args.dout, 'train_sanity.debug_epoch_{}.preds.json'.format(epoch))
            with open(fpred, 'wt') as f:
                json.dump(self.make_debug(p_train_sanity, train_sanity), f, indent=2)
            # time
            time_report['make_debug_train'] += time.time() - start_time

            # save the latest checkpoint
            # time
            start_time = time.time()
            if args.save_every_epoch:
                fsave = os.path.join(args.dout, 'net_epoch_%d.pth' % epoch)
            else:
                fsave = os.path.join(args.dout, 'latest.pth')
            torch.save({
                'metric': stats,
                'model': self.state_dict(),
                'optim': optimizer.state_dict(),
                'args': self.args,
                'vocab': self.vocab,
                'epoch': epoch,
            }, fsave)
            # time
            time_report['torch_save_last'] += time.time() - start_time

            pprint.pprint(stats)

            # time
            time_report['epoch_time'] += time.time() - epoch_start_time

            # time
            for k, v in sorted(time_report.items(), key=lambda x: x[1], reverse=True):
                print('{:<30}{:<40}'.format(k, round(v, 3)))

    def run_pred(self, dev, args=None, name='dev', iter=0, 
                 validate_teacher_forcing=False, validate_sample_output=False):
        '''
        validation loop
        '''
        args = args or self.args
        m_dev = collections.defaultdict(list)
        p_dev = {}
        self.eval()
        total_loss = list()
        dev_iter = iter
        for batch, feat in self.iterate(dev, args.batch):
            out = self.forward(feat, validate_teacher_forcing=validate_teacher_forcing, validate_sample_output=validate_sample_output)
            preds = self.extract_preds(out, batch, feat)
            p_dev.update(preds)
            loss, perplexity = self.compute_loss(out, batch, feat)
            for k, v in loss.items():
                ln = 'batch_loss_' + k
                m_dev[ln].append(v.item())
                if validate_teacher_forcing:
                    self.summary_writer.add_scalar("%s/%s" % (name, ln), v.item(), dev_iter)
            sum_loss = sum(loss.values())
            total_loss.append(float(sum_loss.detach().cpu()))
            m_dev['perplexity'].append(perplexity.item())
            if validate_teacher_forcing:
                self.summary_writer.add_scalar("%s/batch_total_loss" % (name), sum_loss, dev_iter)
                self.summary_writer.add_scalar("%s/batch_perplexity" % (name), perplexity.item(), dev_iter)
            dev_iter += len(batch)

        m_dev = {k: sum(v) / len(v) for k, v in m_dev.items()}
        total_loss = sum(total_loss) / len(total_loss)
        return p_dev, dev_iter, total_loss, m_dev

    def run_eval(self, splits, args=None, epoch=0):
        '''
        eval loop
        '''
        args = args or self.args

        # splits
        train_sanity = splits['train_sanity']
        # ann_0, ann_1 and ann_2 have the same action sequence, only ann_0 is needed for validation
        valid_seen = [t for t in splits['valid_seen'] if t['repeat_idx'] == 0]
        valid_unseen = [t for t in splits['valid_unseen'] if t['repeat_idx'] == 0]       

        # debugging: chose a small fraction of the dataset
        if self.args.dataset_fraction > 0:
            small_valid_size = int((self.args.dataset_fraction * 0.3) / 2)
            train_sanity = train_sanity[:small_valid_size]
            valid_seen = valid_seen[:small_valid_size]
            valid_unseen = valid_unseen[:small_valid_size]

        # debugging: use to check if training loop works without waiting for full epoch
        if self.args.fast_epoch:
            train_sanity = train_sanity[:16]
            valid_seen = valid_seen[:16]
            valid_unseen = valid_unseen[:16]

        # display dout
        print("Saving to: %s" % self.args.dout)

        #-------------------------------------------------------
        # compute metrics for train_sanity, teacher-forcing
        _, _, total_train_sanity_loss, m_train_sanity_teacher = self.run_pred(train_sanity, args=args, name='train_sanity', iter=0, validate_teacher_forcing=True)
        m_train_sanity_teacher['total_loss'] = float(total_train_sanity_loss)
        self.summary_writer.add_scalar('train_sanity/epoch_perplexity_teacher_forcing', m_train_sanity_teacher['perplexity'], epoch)
        self.summary_writer.add_scalar('train_sanity/epoch_total_loss_teacher_forcing', m_train_sanity_teacher['total_loss'], epoch)
             
        # compute metrics for valid_seen, teacher-forcing
        _, _, total_valid_seen_loss, m_valid_seen_teacher = self.run_pred(valid_seen, args=args, name='valid_seen', iter=0, validate_teacher_forcing=True)
        m_valid_seen_teacher['total_loss'] = float(total_valid_seen_loss)
        self.summary_writer.add_scalar('valid_seen/epoch_perplexity_teacher_forcing', m_valid_seen_teacher['perplexity'], epoch)
        self.summary_writer.add_scalar('valid_seen/epoch_total_loss_teacher_forcing', m_valid_seen_teacher['total_loss'], epoch)

        # compute metrics for valid_unseen, teacher-forcing
        _, _, total_valid_unseen_loss, m_valid_unseen_teacher = self.run_pred(valid_unseen, args=args, name='valid_unseen', iter=0, validate_teacher_forcing=True)
        m_valid_unseen_teacher['total_loss'] = float(total_valid_unseen_loss)
        self.summary_writer.add_scalar('valid_unseen/epoch_perplexity_teacher_forcing', m_valid_unseen_teacher['perplexity'], epoch)
        self.summary_writer.add_scalar('valid_unseen/epoch_total_loss_teacher_forcing', m_valid_unseen_teacher['total_loss'], epoch)

        #-------------------------------------------------------
        # compute metrics for train_sanity, student-forcing, argmax
        p_train_sanity_argmax, _, _, m_train_sanity_student_argmax = self.run_pred(train_sanity, args=args, name='train_sanity', iter=0)
        m_train_sanity_student_argmax.update(self.compute_metric(p_train_sanity_argmax, train_sanity))
        self.summary_writer.add_scalar('train_sanity/epoch_BLEU_student_forcing_argmax', m_train_sanity_student_argmax['BLEU'], epoch)
             
        # compute metrics for valid_seen, student-forcing, argmax
        p_valid_seen_argmax, _, _, m_valid_seen_student_argmax = self.run_pred(valid_seen, args=args, name='valid_seen', iter=0)
        m_valid_seen_student_argmax.update(self.compute_metric(p_valid_seen_argmax, valid_seen))
        self.summary_writer.add_scalar('valid_seen/epoch_BLEU_student_forcing_argmax', m_valid_seen_student_argmax['BLEU'], epoch)

        # compute metrics for valid_unseen, student-forcing, argmax
        p_valid_unseen_argmax, _, _, m_valid_unseen_student_argmax = self.run_pred(valid_unseen, args=args, name='valid_unseen', iter=0)
        m_valid_unseen_student_argmax.update(self.compute_metric(p_valid_unseen_argmax, valid_unseen))
        self.summary_writer.add_scalar('valid_unseen/epoch_BLEU_student_forcing_argmax', m_valid_unseen_student_argmax['BLEU'], epoch)

        #-------------------------------------------------------
        # compute metrics for train_sanity, student-forcing, sampled
        p_train_sanity_sampled, _, _, m_train_sanity_student_sampled = self.run_pred(train_sanity, args=args, name='train_sanity', iter=0, validate_sample_output=True)
        m_train_sanity_student_sampled.update(self.compute_metric(p_train_sanity_sampled, train_sanity))
        self.summary_writer.add_scalar('train_sanity/epoch_BLEU_student_forcing_sampled', m_train_sanity_student_sampled['BLEU'], epoch)
             
        # compute metrics for valid_seen, student-forcing, sampled
        p_valid_seen_sampled, _, _, m_valid_seen_student_sampled = self.run_pred(valid_seen, args=args, name='valid_seen', iter=0, validate_sample_output=True)
        m_valid_seen_student_sampled.update(self.compute_metric(p_valid_seen_sampled, valid_seen))
        self.summary_writer.add_scalar('valid_seen/epoch_BLEU_student_forcing_sampled', m_valid_seen_student_sampled['BLEU'], epoch)

        # compute metrics for valid_unseen, student-forcing, sampled
        p_valid_unseen_sampled, _, _, m_valid_unseen_student_sampled = self.run_pred(valid_unseen, args=args, name='valid_unseen', iter=0, validate_sample_output=True)
        m_valid_unseen_student_sampled.update(self.compute_metric(p_valid_unseen_sampled, valid_unseen))
        self.summary_writer.add_scalar('valid_unseen/epoch_BLEU_student_forcing_sampled', m_valid_unseen_student_sampled['BLEU'], epoch)

        #-------------------------------------------------------
        m_train_sanity, m_valid_seen, m_valid_unseen = {}, {}, {}

        m_train_sanity['perplexity'], m_valid_seen['perplexity'], m_valid_unseen['perplexity'] = \
            m_train_sanity_teacher['perplexity'], m_valid_seen_teacher['perplexity'], m_valid_unseen_teacher['perplexity']
        m_train_sanity['total_loss'], m_valid_seen['total_loss'], m_valid_unseen['total_loss'] = \
            m_train_sanity_teacher['total_loss'], m_valid_seen_teacher['total_loss'], m_valid_unseen_teacher['total_loss']
        m_train_sanity['BLEU_argmax'], m_valid_seen['BLEU_argmax'], m_valid_unseen['BLEU_argmax'] = \
            m_train_sanity_student_argmax['BLEU'], m_valid_seen_student_argmax['BLEU'], m_valid_unseen_student_argmax['BLEU']
        m_train_sanity['BLEU_sampled'], m_valid_seen['BLEU_sampled'], m_valid_unseen['BLEU_sampled'] = \
            m_train_sanity_student_sampled['BLEU'], m_valid_seen_student_sampled['BLEU'], m_valid_unseen_student_sampled['BLEU']

        stats = {'epoch': epoch, 'train_sanity': m_train_sanity, 'valid_seen': m_valid_seen, 'valid_unseen': m_valid_unseen}
        #-------------------------------------------------------
        # Make Debugging output for student-forced argmax and sampled decoding

        # train_sanity
        fpred = os.path.join(args.dout, 'train_sanity_argmax.debug_epoch_{}.preds.json'.format(epoch if epoch is not None else ''))
        with open(fpred, 'wt') as f:
            json.dump(self.make_debug(p_train_sanity_argmax, train_sanity), f, indent=2)
        fpred = os.path.join(args.dout, 'train_sanity_sampled.debug_epoch_{}.preds.json'.format(epoch if epoch is not None else ''))
        with open(fpred, 'wt') as f:
            json.dump(self.make_debug(p_train_sanity_sampled, train_sanity), f, indent=2)

        # valid_seen
        fpred = os.path.join(args.dout, 'valid_seen_argmax.debug_epoch_{}.preds.json'.format(epoch if epoch is not None else ''))
        with open(fpred, 'wt') as f:
            json.dump(self.make_debug(p_valid_seen_argmax, valid_seen), f, indent=2)
        fpred = os.path.join(args.dout, 'valid_seen_sampled.debug_epoch_{}.preds.json'.format(epoch if epoch is not None else ''))
        with open(fpred, 'wt') as f:
            json.dump(self.make_debug(p_valid_seen_sampled, valid_seen), f, indent=2)

        # valid_unseen
        fpred = os.path.join(args.dout, 'valid_unseen_argmax.debug_epoch_{}.preds.json'.format(epoch if epoch is not None else ''))
        with open(fpred, 'wt') as f:
            json.dump(self.make_debug(p_valid_unseen_argmax, valid_unseen), f, indent=2)
        fpred = os.path.join(args.dout, 'valid_unseen_sampled.debug_epoch_{}.preds.json'.format(epoch if epoch is not None else ''))
        with open(fpred, 'wt') as f:
            json.dump(self.make_debug(p_valid_unseen_sampled, valid_unseen), f, indent=2)

        #-------------------------------------------------------
        pprint.pprint(stats)

        return m_train_sanity, m_valid_seen, m_valid_unseen

    def featurize(self, batch):
        raise NotImplementedError()

    def forward(self, feat, max_decode=100):
        raise NotImplementedError()

    def extract_preds(self, out, batch, feat):
        raise NotImplementedError()

    def compute_loss(self, out, batch, feat):
        raise NotImplementedError()

    def compute_metric(self, preds, data):
        raise NotImplementedError()

    def get_task_and_ann_id(self, ex):
        '''
        single string for task_id and annotation repeat idx
        '''
        return "%s_%s" % (ex['task_id'], str(ex['repeat_idx']))

    def make_debug(self, preds, data):
        '''
        readable output generator for debugging
        '''
        debug = {}
        for task in data:
            ex = self.load_task_json(task)
            i = self.get_task_and_ann_id(ex)
            debug[i] = {
                # Task Location root
                'root': ex['root'],
                # Input - Low-level actions
                'action_low': [a['discrete_action']['action'] for a in ex['plan']['low_actions']],
                # Input - High-level actions
                'action_high': [a['discrete_action']['action'] for a in ex['plan']['high_pddl']],
                # Predicted - Language
                'p_lang_instr': preds[i]['lang_instr']
                # 'p_action_low': preds[i]['lang_instr'].split(),
            }
        return debug

    def load_task_json(self, task):
        '''
        load preprocessed json from disk
        '''
        json_path = os.path.join(self.args.data, task['task'], '%s' % self.args.pp_folder, 'ann_%d.json' % task['repeat_idx'])
        retry = 0
        while True:
            try:
                if retry > 0:
                    print ('retrying {}'.format(retry))
                with open(json_path) as f:
                    data = json.load(f)
                return data
            except:
                retry += 1
                time.sleep(5)
                pass

    def load_task_jsons(self, task):
        '''
        load all preprocessed jsons with matching task index from disk. 
        do this to gather all 3 versions of language annotations.
        '''
        dataset = []
        for i in range(3):
            json_path = os.path.join(self.args.data, task['task'], '%s' % self.args.pp_folder, 'ann_%d.json' % i)
            if os.path.exists(json_path):
                retry = 0
                while True:
                    try:
                        if retry > 0:
                            print ('retrying {}'.format(retry))
                        with open(json_path) as f:
                            dataset.append(json.load(f))
                        break
                    except:
                        retry += 1
                        time.sleep(5)
                        pass
        return dataset        

    def get_task_root(self, ex):
        '''
        returns the folder path of a trajectory
        '''
        return os.path.join(self.args.data, ex['split'], *(ex['root'].split('/')[-2:]))

    def iterate(self, data, batch_size):
        '''
        breaks dataset into batch_size chunks for training
        '''
        for i in trange(0, len(data), batch_size, desc='batch'):
            tasks = data[i:i+batch_size]
            batch = [self.load_task_json(task) for task in tasks]
            feat = self.featurize(batch)
            yield batch, feat

    def zero_input(self, x, keep_end_token=True):
        '''
        pad input with zeros (used for ablations)
        '''
        end_token = [x[-1]] if keep_end_token else [self.pad]
        return list(np.full_like(x[:-1], self.pad)) + end_token

    def zero_input_list(self, x, keep_end_token=True):
        '''
        pad a list of input with zeros (used for ablations)
        '''
        end_token = [x[-1]] if keep_end_token else [self.pad]
        lz = [list(np.full_like(i, self.pad)) for i in x[:-1]] + end_token
        return lz

    @staticmethod
    def adjust_lr(optimizer, init_lr, epoch, decay_epoch=5):
        '''
        decay learning rate every decay_epoch
        '''
        lr = init_lr * (0.1 ** (epoch // decay_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    @classmethod
    def load(cls, fsave, retain_args=None):
        '''
        load pth model from disk
        '''
        save = torch.load(fsave)

        args = save['args']
        if retain_args is not None:
            args.train_teacher_forcing = True # TODO remove
            args.train_student_forcing_prob = 0. # TODO remove
            args.data = '/root/data_alfred/json_feat_2.1.0'
            args.dout = retain_args.dout

        model = cls(args, save['vocab'])
        model.load_state_dict(save['model'])
        next_epoch = int(save['epoch']) + 1
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer.load_state_dict(save['optim'])
        return model, optimizer, next_epoch

    @classmethod
    def has_interaction(cls, action):
        '''
        check if low-level action is interactive
        '''
        non_interact_actions = ['MoveAhead', 'Rotate', 'Look', '<<stop>>', '<<pad>>', '<<seg>>']
        if any(a in action for a in non_interact_actions):
            return False
        else:
            return True
