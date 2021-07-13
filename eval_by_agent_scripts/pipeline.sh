# Set up paths
# TODO sub toy data for now

export ALFRED_ROOT=~/alfred
export DATA_ROOT=/media/legg/seagate-2tb-hdd
export SPLITS_DIR=$DATA_ROOT/splits/
export ALFRED_DATA=$DATA_ROOT/json_dummy/
export MODEL_SAVE=$DATA_ROOT/exp/
export OBJECT_VOCAB=/media/legg/seagate-2tb-hdd/unlabeled_12k_20201206/seen/objects_20200522.object_vocab


#####################################################################################
# 1. Train Natural Language Instruction Giving Models (Explainer or Seq2Seq Baseline)

export ALFRED_UNFILTERED_SPLIT=$SPLITS_DIR/?_raw.json
export ALFRED_FILTERED_SPLIT=$SPLITS_DIR/?_filtered.json
export LM_MODEL=seq2seq_per_subgoal
export LM_SAVE_NAME=model:$LM_MODEL,name:subgoal_level_obj_instance_enc_max_pool_dec_aux_loss_weighted_bce_1to2
export LM_DOUT=$MODEL_SAVE/$LM_SAVE_NAME

export GOAL_LM_MODEL=seq2seq_nl_with_frames
export GOAL_LM_SAVE_NAME=model:$LM_MODEL,name:goal_level
export GOAL_LM_DOUT=$MODEL_SAVE/$GOAL_LM_SAVE_NAME

# 1.1 Collect Object States for Original Alfred Data
python gen/scripts/collect_object_states.py --data $ALFRED_DATA --raw_splits $ALFRED_UNFILTERED_SPLIT --in_parallel --num_threads 2

# 1.2 Preprocess Object States into Data Structure for Original Alfred Data
python gen/scripts/featurize_object_states.py --data $ALFRED_DATA --splits $ALFRED_FILTERED_SPLIT --object_vocab $OBJECT_VOCAB

# 1.3 fix action and object states subgoal alignment
python gen/scripts/fix_modality_alignments.py --pre_auto_labeling --data $ALFRED_DATA --splits $ALFRED_FILTERED_SPLIT

# 1.4 Train Subgoal-level Langauge Model (Explainer / Baseline)
# Use --preprocess to preprocess ALFRED_DATA if training language model for the first time
python models/train/train_seq2seq.py --model $LM_MODEL --dout $LM_DOUT -data $ALFRED_DATA --splits $ALFRED_FILTERED_SPLIT --gpu --epoch 40 --save_every_epoch --encoder_addons max_pool_obj --decoder_addons aux_loss --object_repr instance --object_vocab $OBJECT_VOCAB --preprocess --pp_folder $LM_SAVE_NAME

# 1.5 Train Goal-level Langauge Model (Seq2Seq)
python models/train/train_seq2seq.py --predict_goal_level_instruction --model $GOAL_LM_MODEL --dout $GOAL_LM_DOUT -data $ALFRED_DATA --splits $ALFRED_UNFILTERED_SPLIT --gpu --epoch 40 --save_every_epoch --preprocess --pp_folder $GOAL_LM_SAVE_NAME


#####################################################################################
# 2. Evaluate Natural Language Instruction Giving Models (Subgoal level)

# 2.1 Select the best model based on tensorboard plots (using BLEU, F1 etc)
export VALID_SEEN_PREDICTIONS=$LM_DOUT/valid_seen.debug_epoch_32.preds.json
export VALID_UNSEEN_PREDICTIONS=$LM_DOUT/valid_unseen.debug_epoch_32.preds.json

# 2.2 Evaluate the corresponding valid split predictions
python models/eval/eval_linguistics.py --valid_seen_predictions_path $VALID_SEEN_PREDICTIONS --valid_unseen_predictions_path $VALID_UNSEEN_PREDICTIONS


#####################################################################################
# 3. Sample and Label New Trajectories

export DT=$(date '+%d%m%Y')
export SAMPLED_SUBDIR=sampled/new_trajectories_T$DT/
export CLEANED_SUBDIR=cleaned/new_trajectories_T$DT/
export PREVIOUSLY_FAILED_TASKS=$ALFRED_ROOT/gen/scripts/task_names.txt
export NEW_SPLIT_FILENAME=sampled_partial_T$DT
# explainer/baseline LM checkpoint used to autolabel new trajectories
export LM_CKPT=$LM_OUT/$LM_SAVE_NAME/net_epoch_??.pth
export LM_TAG=explainer
export GOAL_LM_CKPT=$GOAL_LM_DOUT/$GOAL_LM_SAVE_NAME/net_epoch_??.pth

# 3.1 sample all failed task from alfred/gen/scripts/task_names.txt
python gen/scripts/sample_augmentation_trajectories.py --task_names_path $PREVIOUSLY_FAILED_TASKS --data_root $DATA_ROOT --save_subdir $SAMPLED_SUBDIR --splits_dir $SPLITS_DIR --trials_before_fail 2 --in_parallel --num_processes 3

# 3.2 clean up sampled trajectories
python gen/scripts/cleanup_sampled_trajectories.py --task_names_path $PREVIOUSLY_FAILED_TASKS --data_root $DATA_ROOT --samp_dir $SAMPLED_SUBDIR --clean_subdir $CLEANED_SUBDIR --splits_dir $SPLITS_DIR --new_split_filename "${NEW_SPLIT_FILENAME}_raw.json"

# 3.3 extract resnet features for visual frames
python models/utils/extract_resnet.py --data $DATA_ROOT/$CLEANED_SUBDIR --skip_existing --gpu

# 3.4 collect Object States for New Data
python gen/scripts/collect_object_states.py --data $DATA_ROOT/$CLEANED_SUBDIR --raw_splits $SPLITS_DIR/"${NEW_SPLIT_FILENAME}_raw.json" --in_parallel --num_threads 2

# 3.5 preprocess Object States into Data Structure for New Data
python gen/scripts/featurize_object_states.py --data $DATA_ROOT/$CLEANED_SUBDIR --splits $SPLITS_DIR/"${NEW_SPLIT_FILENAME}_filtered.json" --object_vocab $OBJECT_VOCAB

# 3.6 preprocess action tokens
pythpn data/preprocess.py --action_tokens_only_for_instruction_labeling --data $DATA_ROOT/$CLEANED_SUBDIR --splits $SPLITS_DIR/"${NEW_SPLIT_FILENAME}_filtered.json" --model_path $LM_CKPT --model_name $LM_SAVE_NAME --lmtag $LM_TAG

# 3.7 fix action and object states subgoal alignment
python gen/scripts/fix_modality_alignments.py --pre_auto_labeling --data $DATA_ROOT/$CLEANED_SUBDIR --splits $SPLITS_DIR/"${NEW_SPLIT_FILENAME}_filtered.json"

# 3.8 Language models label the new trajectories
python models/autolabel/explain_trajectories.py --data $DATA --splits $SPLITS --subgoal_level_explainer_checkpt_path $LM_CKPT --goal_level_explainer_checkpt_path $GOAL_LM_CKPT --gpu --lmtag explainer

# 3.9 clean up predictions to remove extra subgoals
python gen/scripts/fix_modality_alignments.py --data $DATA_ROOT/$CLEANED_SUBDIR --splits $SPLITS_DIR/"${NEW_SPLIT_FILENAME}_aligned.json" --lm_tags explainer


#####################################################################################
# 4. Train Data Augmented Alfred Agent

export AGENT_MODEL=seq2seq_im_mask
export AGENT_SAVE_NAME=model:$AGENT_MODEL,name:augmented_by_$LM_TAG
export AGENT_DOUT=$MODEL_SAVE/$AGENT_SAVE_NAME

# 4.1 preprocess sampled trajectories of autolabeled data for agent training
# copy of vocab will be saved to $DATA_ROOT/$CLEANED_SUBDIR/pp_$AGENT_SAVE_NAME
pythpn data/preprocess.py --action_lang_tokens_for_agent_training --data $DATA_ROOT/$CLEANED_SUBDIR --splits $SPLITS_DIR/"${NEW_SPLIT_FILENAME}_aligned.json" --model_name $AGENT_SAVE_NAME

# 4.2 curate the splits manually, e.g. by merging the existing alfred split with augmentation split.
export CURATED_SPLITS=?

# 4.3 train agent
# Use --preprocess to preprocess ALFRED_DATA if training agent for the first time
python models/train/train_seq2seq_agent.py --data $ALFRED_DATA --model $AGENT_MODEL --dout $AGENT_DOUT --splits $CURATED_SPLITS --preprocess --pp_folder pp_$AGENT_SAVE_NAME --gpu --batch 8 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1 --use_augmentation --augmentation_data $DATA_ROOT/$CLEANED_SUBDIR --augmentation_lang_model $LM_TAG --save_every_epoch --epoch 40


#####################################################################################
# 5. Evaluate Data Augmented Alfred Agent

# 5.1 Rank agent training results. Select best epoch by val loss, F1 or exact match
python models/eval/rank_agent_training_results.py --exp_dir $MODEL_SAVE --mod_name $AGENT_SAVE_NAME

# 5.2 Eval by task (valid seen or unseen)
export AGENT_CKPT=$AGENT_DOUT/$AGENT_SAVE_NAME/net_epoch_?.pth
export EVAL_SPLITS=$SPLITS_DIR/oct21.json
export EVAL_SPLIT=valid_unseen
python models/eval/eval_seq2seq_agent.py --model_path $AGENT_CKPT --data $DATA --splits $EVAL_SPLITS --eval_split $EVAL_SPLIT --model models.model.seq2seq_im_mask --gpu --num_threads 3

# 5.3 Eval by subgoal type
python models/eval/eval_seq2seq_agent.py --model_path $AGENT_CKPT --data $DATA --splits $EVAL_SPLITS --eval_split $EVAL_SPLIT --model models.model.seq2seq_im_mask --gpu --num_threads 3 --subgoals all