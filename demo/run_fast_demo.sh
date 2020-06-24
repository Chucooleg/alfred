export ALFRED_ROOT=/root/data/home/hoyeung/alfred
export DATA=/root/data_alfred/json_feat_2.1.0
export EVAL_DATA_ROOT=/root/data_alfred/demo_generated/
export SPLITS_ROOT=/root/data_alfred/splits/
export MODEL_DIR=/root/data/home/hoyeung/blob_alfred_data/exp_all/
export EXPLAINER=$MODEL_DIR/model:seq2seq_per_subgoal,name:v2_epoch_40_obj_instance_enc_max_pool_dec_aux_loss_weighted_bce_1to2/net_epoch_32.pth
export GOAL_EXPLAINER=$MODEL_DIR/model:seq2seq_nl_with_frames,name:v1.5_epoch_50_high_level_instrs/net_epoch_10.pth
export ALFRED_AGENT=$MODEL_DIR/pretrained_model/model:seq2seq_im_mask,name:base30_pm010_sg010_01/best_seen.pth


cd $ALFRED_ROOT
# query task tuple and make new split
python demo/query_and_make_split.py

# extract lastest split file name
export SPLITS_FILE=$(ls $SPLITS_ROOT | grep -i 'demo_*' | tail -n 1)
export SPLITS=$SPLITS_ROOT/$SPLITS_FILE

# explainer explain!
cd $ALFRED_ROOT
python models/run_demo/explain_fast_demo_trajectories.py --data $DATA --splits $SPLITS --low_level_explainer_checkpt_path $EXPLAINER --high_level_explainer_checkpt_path $GOAL_EXPLAINER --gpu


export EVAL_DATA_FILE=$(ls $EVAL_DATA_ROOT | grep -i 'new_trajectories*' | tail -n 1)
export EVAL_DATA=$EVAL_DATA_ROOT/$EVAL_DATA_FILE

# eval alfred
cd $ALFRED_ROOT/
python models/eval/eval_seq2seq.py --model_path $ALFRED_AGENT --data $EVAL_DATA --splits $SPLITS --model models.model.seq2seq_im_mask --gpu --preprocess --demo_mode

python models/eval/eval_seq2seq.py --model_path $ALFRED_AGENT --data $EVAL_DATA --splits $SPLITS --model models.model.seq2seq_im_mask --gpu --demo_mode --subgoals PickupObject

python models/eval/eval_seq2seq.py --model_path $ALFRED_AGENT --data $EVAL_DATA --splits $SPLITS --model models.model.seq2seq_im_mask --gpu --demo_mode --subgoals GotoLocation

echo 'Pipeline finihsed.'