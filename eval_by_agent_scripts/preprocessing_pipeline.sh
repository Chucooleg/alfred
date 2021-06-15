# STEP 0
# Use alfred/gen/scripts/sample_augmentation_trajectories_uniform.py to sample all failed task from alfred/gen/scripts/task_names.txt (originally from fails.txt from Mohit)

# STEP 1
# use alfred/Cleanup_sampled_fail_trajectories.ipynb to clean up collected trajectories and save the raw splits.

# STEP 2
# Manually copy the data back to /data_alfred from /home/hoyeung/json_data_augmentation_20200819 if so.
# cp -r /home/hoyeung/json_data_augmentation_20200819 /data_alfred/json_data_augmentation_20200819
# !!!TODO we need to move the images as well!

export DATA_ROOT=/root/data_alfred
export ALFRED_ROOT=/root/data/home/hoyeung/alfred

# STEP 3
echo "-------------------------------------------------------------------------------------------------------------------------"
echo "3. extract resnet features"

export DATA=$DATA_ROOT/json_data_augmentation_20200820/
# or
# export DATA=$DATA_ROOT/demo_generated/new_trajectories_dummy/
echo "DATA is ${DATA}"
cd $ALFRED_ROOT/
# python models/utils/extract_resnet.py --data $DATA --skip_existing
python models/utils/extract_resnet.py --data $DATA --skip_existing --gpu 


# STEP 4 TODO NOTE probably need to turn off assert success
echo "-------------------------------------------------------------------------------------------------------------------------"
echo "4. planned trajectory check and extract object state features"

# CHANGE!!
export DATA=$DATA_ROOT/json_dummy/
export RAW_SPLITS=$DATA_ROOT/splits/sample_failed_dummy_raw.json
export SPLITS=$DATA_ROOT/splits/sample_failed_dummy.json


export DATA=$DATA_ROOT/json_data_augmentation_20200820/
export RAW_SPLITS=$DATA_ROOT/splits/sample_failed_20200820_raw.json
export SPLITS=$DATA_ROOT/splits/sample_failed_20200820.json
echo "RAW SPLIT IS ${RAW_SPLITS}"
echo "EXPECTED OUTPUT SPLIT IS ${SPLITS}"

cd $ALFRED_ROOT/gen
# TODO NOTE probably need to turn off assert success
python scripts/collect_demo_object_states.py --data $DATA --raw_splits $RAW_SPLITS --in_parallel --num_threads 2
# python scripts/collect_demo_object_states.py --data $DATA --raw_splits $RAW_SPLITS --first_task_only

python models/utils/extract_resnet.py --data $DATA --skip_existing
# STEP 5
echo "-------------------------------------------------------------------------------------------------------------------------"
echo "5. preprocess action tokens"

export DATA=$DATA_ROOT/json_data_augmentation_20200820/
export RAW_SPLITS=$DATA_ROOT/splits/sample_failed_20200820_raw.json
export SPLITS=$DATA_ROOT/splits/sample_failed_20200820.json

export MODEL_DIR=/root/data/home/hoyeung/blob_alfred_data/exp_all/
export EXPLAINER=$MODEL_DIR/model:seq2seq_per_subgoal,name:v2_epoch_40_obj_instance_enc_max_pool_dec_aux_loss_weighted_bce_1to2/net_epoch_32.pth
export BASELINE=$MODEL_DIR/model:seq2seq_per_subgoal,name:v2_epoch_35_baseline/net_epoch_29.pth
export GOAL_EXPLAINER=$MODEL_DIR/model:seq2seq_nl_with_frames,name:v1.5_epoch_50_high_level_instrs/net_epoch_10.pth

cd $ALFRED_ROOT/
python data/preprocess_demo_trajectories.py --data $DATA --splits $SPLITS --explainer_path $EXPLAINER
python data/preprocess_demo_trajectories.py --data $DATA --splits $SPLITS --explainer_path $BASELINE
python data/preprocess_demo_trajectories.py --data $DATA --splits $SPLITS --explainer_path $GOAL_EXPLAINER --high_level_goal_explainer


# STEP 6
echo "-------------------------------------------------------------------------------------------------------------------------"
echo "6. preprocess object state features"
cd $ALFRED_ROOT/
python data/preprocess_demo_object_states.py --data $DATA --splits $SPLITS --explainer_path $EXPLAINER  # BASELINE/GOAL_EXPLAINER doesn't need this step


# STEP 6
echo "-------------------------------------------------------------------------------------------------------------------------"
echo "6.5 filter out shape mismatch for Data Augmentation"
# Filter_out_shape_mismatch_for_Data_Augmentation.ipynb
# (Pre Explainer/Baseline Prediction)


# STEP 7 Where do the predicted files go?
echo "-------------------------------------------------------------------------------------------------------------------------"
echo "7. explainer and baseline predicts instructions"
cd $ALFRED_ROOT/

export SPLITS=$DATA_ROOT/splits/sample_failed_20200820_filtered.json

# first, move data under the split folder ?
python models/run_demo/explain_full_demo_trajectories.py --data $DATA --splits $SPLITS --low_level_explainer_checkpt_path $EXPLAINER --high_level_explainer_checkpt_path $GOAL_EXPLAINER --gpu
python models/run_demo/explain_full_demo_trajectories.py --data $DATA --splits $SPLITS --low_level_explainer_checkpt_path $BASELINE --high_level_explainer_checkpt_path $GOAL_EXPLAINER --gpu --baseline


python models/run_demo/explain_full_demo_trajectories.py --data $DATA --splits $SPLI
TS --low_level_explainer_checkpt_path $EXPLAINER --high_level_explainer_checkpt_path $GOAL_EXPLAINER --gpu --fast_epoch


# STEP 8
echo "-------------------------------------------------------------------------------------------------------------------------"
echo "8. Clean up predicted instructions to remove extra subgoals"
# Filter_out_shape_mismatch_for_Data_Augmentation.ipynb
# (Post Explainer/Baseline Prediction)


# STEP 9
echo "-------------------------------------------------------------------------------------------------------------------------"
echo "9. Running preprocessing for agent training"
# Preprocess_sampled_instructions_for_augmentation_20200825.ipynb


# STEP 10
echo "-------------------------------------------------------------------------------------------------------------------------"
echo "10. Put the splits together"
# Preprocess_sampled_instructions_for_augmentation_20200825.ipynb

# STEP 10.5
echo "-------------------------------------------------------------------------------------------------------------------------"
echo "10.5 Filter away unseen envs from trainingr"
# Preprocess_sampled_instructions_for_augmentation_20210608.ipynb

# STEP 11 - 13
# see agent_train_eval.sh