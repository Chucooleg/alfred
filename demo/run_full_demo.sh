export ALFRED_ROOT=/root/data/home/hoyeung/alfred
export DATA_DIR=new_trajectories/
export DATA_ROOT=/root/data_alfred/demo_generated/
export SPLITS_ROOT=/root/data_alfred/splits/
export MODEL_DIR=/root/data/home/hoyeung/blob_alfred_data/exp_all/
export EXPLAINER=$MODEL_DIR/model:seq2seq_per_subgoal,name:v2_epoch_40_obj_instance_enc_max_pool_dec_aux_loss_weighted_bce_1to2/net_epoch_32.pth
export GOAL_EXPLAINER=$MODEL_DIR/model:seq2seq_nl_with_frames,name:v1.5_epoch_50_high_level_instrs/net_epoch_10.pth
export ALFRED_AGENT=$MODEL_DIR/pretrained_model/model:seq2seq_im_mask,name:base30_pm010_sg010_01/best_seen.pth


echo "-------------------------------------------------------------------------------------------------------------------------"
echo "1. planner collect trajectory according to task tuple"
cd $ALFRED_ROOT/gen
python scripts/sample_demo_trajectories.py --goal look_at_obj_in_light --pickup AlarmClock --movable None --receptacle DeskLamp --scene 314 --save_path $DATA_ROOT/$DATA_DIR --in_parallel --num_threads 5 --repeats_per_cond 1 --trials_before_fail 5

# exit if sampling python script exits with error
ret=$?
if [ $ret -ne 0 ]; then
     echo 'Exiting pipeline.'
     exit 1
fi

# T20200621_233508_457342
export TIMESTAMP=$(ls /root/data_alfred/demo_generated/ | grep -i 'new_*' | tail -n 1 | grep -oP "T.*")
# /root/data_alfred/demo_generated/new_trajectories_T20200621_233508_457342/
export DATA=$DATA_ROOT/$(ls /root/data_alfred/demo_generated/ | grep -i 'new_*' | tail -n 1)/
echo "DATA is ${DATA}"
echo "-------------------------------------------------------------------------------------------------------------------------"
echo "2. extract resnet features"
cd $ALFRED_ROOT/
python models/utils/extract_resnet.py --data $DATA --gpu

# /root/data_alfred/splits/demo_T20200621_233508_457342_raw.json
export RAW_SPLITS=$SPLITS_ROOT/demo_${TIMESTAMP}_raw.json
# /root/data_alfred/splits/demo_T20200621_233508_457342.json
export SPLITS=$SPLITS_ROOT/demo_${TIMESTAMP}.json

echo "-------------------------------------------------------------------------------------------------------------------------"
echo "3. planned trajectory check and extract object state features"
cd $ALFRED_ROOT/gen
python scripts/collect_demo_object_states.py --data $DATA --raw_splits $RAW_SPLITS --first_task_only

echo "-------------------------------------------------------------------------------------------------------------------------"
echo "4. preprocess action tokens"
cd $ALFRED_ROOT/
python data/preprocess_demo_trajectories.py --data $DATA --splits $SPLITS --explainer_path $EXPLAINER
python data/preprocess_demo_trajectories.py --data $DATA --splits $SPLITS --explainer_path $GOAL_EXPLAINER --high_level_goal_explainer

echo "-------------------------------------------------------------------------------------------------------------------------"
echo "5. preprocess object state features"
cd $ALFRED_ROOT/
python data/preprocess_demo_object_states.py --data $DATA --splits $SPLITS --explainer_path $EXPLAINER

echo "-------------------------------------------------------------------------------------------------------------------------"
echo "6. explainer predicts instructions"
cd $ALFRED_ROOT/
python models/run_demo/explain_demo_trajectories.py --data $DATA --splits $SPLITS --low_level_explainer_checkpt_path $EXPLAINER --high_level_explainer_checkpt_path $GOAL_EXPLAINER --gpu


echo "-------------------------------------------------------------------------------------------------------------------------"
echo "7. eval explainer on alfred"
cd $ALFRED_ROOT/
python models/eval/eval_seq2seq.py --model_path $ALFRED_AGENT --eval_split demo --data $DATA --splits $SPLITS --model models.model.seq2seq_im_mask --gpu --preprocess --demo_mode

python models/eval/eval_seq2seq.py --model_path $ALFRED_AGENT --eval_split demo --data $DATA --splits $SPLITS --model models.model.seq2seq_im_mask --gpu --demo_mode --subgoals PickupObject

python models/eval/eval_seq2seq.py --model_path $ALFRED_AGENT --eval_split demo --data $DATA --splits $SPLITS --model models.model.seq2seq_im_mask --gpu --demo_mode --subgoals GotoLocation



# 1.1 Python Query Task Tuple Interactively (Terminal)
# 1.2 Python Make split (save out to split file)
# 1.3 Python print Video URL link
# 2.1.Script Call Explainers, Python run explainers
# 3.1 Script Call Python to Launch tkinter to collect human traj
# 4.1 Alfred Eval
# 5.1 Human Eval

#- Python query task tuple interactively 
#- Build Task Tuple Look up
#- main call that makes a split
#- print video url
#- run explainer at the same time
#- Launch tkinter collect human traj

#- Eval human traj