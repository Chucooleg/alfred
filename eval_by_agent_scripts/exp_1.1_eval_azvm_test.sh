echo 'WIthin Exp 1 Sub Script'

# Set output Directory
export MODEL_NAME=experiment_1_original_train_set_plus_all_failures_annotated_by_$AUGMENTATION_LANG_MODEL
export DOUT=$DATA_ROOT/exp/model:$MODEL,name:$MODEL_NAME

echo $MODEL_NAME
echo $DOUT

# # Training
cd $ALFRED_ROOT
# echo Start training agent $MODEL_NAME
# echo Will Save to $DOUT...
# python -m memory_profiler models/train/train_seq2seq_agent.py --data $DATA --model $MODEL --dout $DOUT --splits $SPLITS --pp_folder $PP --gpu --batch 8 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1 --save_every_epoch --epoch 27 --resume $DOUT/net_epoch_19.pth  --lr 1e-5

# Eval
export EVAL_SPLITS=$DATA_ROOT/splits/oct21.json

# Eval on Test
echo 'Start Validation on Test'
export AGENT_MODEL=$DOUT/net_epoch_23.pth
ls -l
python models/eval/leaderboard.py --model_path $AGENT_MODEL --data $DATA --splits $EVAL_SPLITS --model models.model.seq2seq_im_mask --gpu --num_threads 5

