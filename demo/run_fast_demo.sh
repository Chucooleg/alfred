export DATA_ROOT=$1
export DATA=$2
export ALFRED_ROOT=$3
export MODEL_DIR=$4
export USE_GPU=$5

export DATA=$DATA_ROOT/$DATA
export EVAL_DATA_ROOT=$DATA_ROOT/demo_generated/
export SPLITS_ROOT=$DATA_ROOT/splits/
export EXPLAINER=$MODEL_DIR/model:seq2seq_per_subgoal,name:v2_epoch_40_obj_instance_enc_max_pool_dec_aux_loss_weighted_bce_1to2/net_epoch_32.pth
export GOAL_EXPLAINER=$MODEL_DIR/model:seq2seq_nl_with_frames,name:v1.5_epoch_50_high_level_instrs/net_epoch_10.pth
export ALFRED_AGENT=$MODEL_DIR/pretrained_model/model:seq2seq_im_mask,name:base30_pm010_sg010_01/best_seen.pth


cd $ALFRED_ROOT
# query task tuple and make new split
python demo/query_and_make_split.py --data $DATA --splits $SPLITS_ROOT

# extract lastest split file name
export SPLITS_FILE=$(ls $SPLITS_ROOT | grep -i 'demo_*' | tail -n 1)
export SPLITS=$SPLITS_ROOT/$SPLITS_FILE

# explainer explain!
cd $ALFRED_ROOT
if [ $USE_GPU == 'gpu' ] ; then
    echo 'Using gpu'
  python models/run_demo/explain_fast_demo_trajectories.py --data $DATA --dout $EVAL_DATA_ROOT --splits $SPLITS --low_level_explainer_checkpt_path $EXPLAINER --high_level_explainer_checkpt_path $GOAL_EXPLAINER --gpu
else
    echo 'Not using gpu'
  python models/run_demo/explain_fast_demo_trajectories.py --data $DATA --dout $EVAL_DATA_ROOT --splits $SPLITS --low_level_explainer_checkpt_path $EXPLAINER --high_level_explainer_checkpt_path $GOAL_EXPLAINER
fi

export EVAL_DATA_FILE=$(ls $EVAL_DATA_ROOT | grep -i 'new_trajectories*' | tail -n 1)
export EVAL_DATA=$EVAL_DATA_ROOT/$EVAL_DATA_FILE

# eval alfred
# cd $ALFRED_ROOT/
# python models/eval/eval_seq2seq.py --model_path $ALFRED_AGENT --data $EVAL_DATA --splits $SPLITS --model models.model.seq2seq_im_mask --gpu --preprocess --demo_mode
# python models/eval/eval_seq2seq.py --model_path $ALFRED_AGENT --data $EVAL_DATA --splits $SPLITS --model models.model.seq2seq_im_mask --gpu --demo_mode --subgoals PickupObject
# python models/eval/eval_seq2seq.py --model_path $ALFRED_AGENT --data $EVAL_DATA --splits $SPLITS --model models.model.seq2seq_im_mask --gpu --demo_mode --subgoals GotoLocation

printf 'SPLITS', $SPLITS
echo 'EVAL DATA', $EVAL_DATA

cd $ALFRED_ROOT
python interface.py --split $SPLITS --data $EVAL_DATA
# python interface.py --split /root/data_alfred/splits//demo_T20200624_162008_674795.json --data /root/data_alfred/demo_generated//new_trajectories_T20200624_162008_674795 --window_size 900 &

printf 'Pipeline finihsed.'