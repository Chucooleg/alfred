echo 'WIthin Exp 1 Sub Script'

echo 'AUGMENTATION_LANG_MODEL:' $AUGMENTATION_LANG_MODEL

# Set output Directory
export MODEL_NAME=experiment_1_original_train_set_plus_12k_successes_annotated_by_$AUGMENTATION_LANG_MODEL
export DOUT=$DATA_ROOT/exp/model:$MODEL,name:$MODEL_NAME

# Training
cd $ALFRED_ROOT
echo Start training agent $MODEL_NAME
echo Will Save to $DOUT...
# python -m memory_profiler models/train/train_seq2seq_agent.py --data $DATA --model $MODEL --dout $DOUT --splits $SPLITS --pp_folder $PP --batch 8 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1 --use_augmentation --augmentation_data $AUGMENTATION_DATA --augmentation_lang_model $AUGMENTATION_LANG_MODEL --save_every_epoch --epoch 30 --lr 1e-5
python -m memory_profiler models/train/train_seq2seq_agent.py --data $DATA --model $MODEL --dout $DOUT --splits $SPLITS --pp_folder $PP --gpu --batch 8 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1 --use_augmentation --augmentation_data $AUGMENTATION_DATA --augmentation_lang_model $AUGMENTATION_LANG_MODEL --save_every_epoch --epoch 30 --lr 1e-5