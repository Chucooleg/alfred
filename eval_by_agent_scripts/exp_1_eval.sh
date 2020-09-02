echo 'WIthin Exp 1 Sub Script'

# Set output Directory
export MODEL_NAME=experiment_1_original_train_set_plus_all_failures_annotated_by_$AUGMENTATION_LANG_MODEL
export DOUT=$DATA_ROOT/exp/model:$MODEL,name:$MODEL_NAME

# Training
cd $ALFRED_ROOT
# echo Start training agent $MODEL_NAME
# echo Will Save to $DOUT...
# python -m memory_profiler models/train/train_seq2seq_agent.py --data $DATA --model $MODEL --dout $DOUT --splits $SPLITS --pp_folder $PP --gpu --batch 8 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1 --use_augmentation --augmentation_data $AUGMENTATION_DATA --augmentation_lang_model $AUGMENTATION_LANG_MODEL --save_every_epoch --epoch 20

# Eval
export EVAL_SPLITS=$DATA_ROOT/splits/oct21.json
echo Eval Split file $EVAL_SPLITS

# Eval on Validation Seen
echo 'Start Validation on Seen'
export AGENT_MODEL=$DOUT/best_seen.pth
export EVAL_SPLIT=valid_seen
python models/eval/eval_seq2seq_agent.py --model_path $AGENT_MODEL --data $DATA --splits $EVAL_SPLITS --eval_split $EVAL_SPLIT --model models.model.seq2seq_im_mask --gpu --num_threads 3 --fast_epoch | tee $DOUT/task_eval_valid_seen.log

# Eval on Validation Seen - Per Subgoal
echo 'Start Validation on Seen - Per Subgoal'
python models/eval/eval_seq2seq_agent.py --model_path $AGENT_MODEL --data $DATA --splits $EVAL_SPLITS --eval_split $EVAL_SPLIT --model models.model.seq2seq_im_mask --gpu --num_threads 3 --subgoals all --fast_epoch | tee $DOUT/subgoal_eval_valid_seen.log


# Eval on Validation Unseen
echo 'Start Validation on Unseen'
export AGENT_MODEL=$DOUT/best_unseen.pth
export EVAL_SPLIT=valid_unseen
python models/eval/eval_seq2seq_agent.py --model_path $AGENT_MODEL --data $DATA --splits $EVAL_SPLITS --eval_split $EVAL_SPLIT --model models.model.seq2seq_im_mask --gpu --num_threads 3 --fast_epoch | tee $DOUT/task_eval_valid_unseen.log

# Eval on Validation Unseen - Per Subgoal
echo 'Start Validation on Unseen - Per Subgoal'
python models/eval/eval_seq2seq_agent.py --model_path $AGENT_MODEL --data $DATA --splits $EVAL_SPLITS --eval_split $EVAL_SPLIT --model models.model.seq2seq_im_mask --gpu --num_threads 3 --subgoals all --fast_epoch | tee $DOUT/subgoal_eval_valid_unseen.log
