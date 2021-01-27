echo 'WIthin Exp 1 Sub Script'

echo 'AUGMENTATION_LANG_MODEL:' $AUGMENTATION_LANG_MODEL
echo 'SEEN_EPOCH:' $SEEN_EPOCH
echo 'UNSEEN_EPOCH:' $UNSEEN_EPOCH


# Set output Directory
export MODEL_NAME=experiment_1_original_train_set_plus_12k_successes_annotated_by_$AUGMENTATION_LANG_MODEL
export DOUT=$DATA_ROOT/exp/model:$MODEL,name:$MODEL_NAME

echo $MODEL_NAME
echo $DOUT

cd $ALFRED_ROOT

# Eval
export EVAL_SPLITS=$DATA_ROOT/splits/oct21.json

# # Eval on Validation Seen
# echo 'Start Validation on Seen model epoch' $SEEN_EPOCH
# export AGENT_MODEL=$DOUT/net_epoch_$SEEN_EPOCH.pth
# export EVAL_SPLIT=valid_seen
# python models/eval/eval_seq2seq_agent.py --model_path $AGENT_MODEL --data $DATA --splits $EVAL_SPLITS --eval_split $EVAL_SPLIT --model models.model.seq2seq_im_mask --gpu --num_threads 8 | tee $DOUT/EvalLog_$AUGMENTATION_LANG_MODEL:seen_epoch$SEEN_EPOCH.txt

# # Eval on Validation Seen - Per Subgoal
# echo 'Start Validation on Seen - Per Subgoal - model epoch' $SEEN_EPOCH
# python models/eval/eval_seq2seq_agent.py --model_path $AGENT_MODEL --data $DATA --splits $EVAL_SPLITS --eval_split $EVAL_SPLIT --model models.model.seq2seq_im_mask --gpu --num_threads 8 --subgoals all | tee $DOUT/EvalLog_$AUGMENTATION_LANG_MODEL:seen_persubgoal_epoch$SEEN_EPOCH.txt


# Eval on Validation Unseen
echo 'Start Validation on Unseen model epoch' $UNSEEN_EPOCH
export AGENT_MODEL=$DOUT/net_epoch_$UNSEEN_EPOCH.pth
export EVAL_SPLIT=valid_unseen
python models/eval/eval_seq2seq_agent.py --model_path $AGENT_MODEL --data $DATA --splits $EVAL_SPLITS --eval_split $EVAL_SPLIT --model models.model.seq2seq_im_mask --gpu --num_threads 8 | tee $DOUT/EvalLog_$AUGMENTATION_LANG_MODEL:unseen_epoch$UNSEEN_EPOCH.txt

# Eval on Validation Unseen - Per Subgoal
echo 'Start Validation on Unseen - Per Subgoal - model epoch' $UNSEEN_EPOCH
python models/eval/eval_seq2seq_agent.py --model_path $AGENT_MODEL --data $DATA --splits $EVAL_SPLITS --eval_split $EVAL_SPLIT --model models.model.seq2seq_im_mask --gpu --num_threads 8 --subgoals all | tee $DOUT/EvalLog_$AUGMENTATION_LANG_MODEL:unseen_persubgoal_epoch$UNSEEN_EPOCH.txt
