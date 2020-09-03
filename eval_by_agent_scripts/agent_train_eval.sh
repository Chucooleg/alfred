# First, Run tmux into the container
# tmux attach -t container_2

# cd /home/legg/alfred/eval_by_agent_scripts/

# General Path Setup
export ENV_HOME=/root/home/hoyeung
export ALFRED_ROOT=$ENV_HOME/alfred
export DATA_ROOT=/root/data_alfred
# export BLOB_ROOT=$ENV_HOME/blob_alfred_data/ # for backup to container
# export BLOB_EXP_DIR=$BLOB_ROOT/exp_all/
export DATA=$DATA_ROOT/json_feat_2.1.0_backup_20200826_agent_training
export PP=pp
export MODEL=seq2seq_im_mask
export AUGMENTATION_DATA=$DATA_ROOT/json_data_augmentation_20200820

# where this script is
export SCRIPT_HOME=$ALFRED_ROOT/eval_by_agent_scripts/
cd $SCRIPT_HOME



echo '--------------------------------------------------------------------------------------'
echo 'Start Experiment 1 - Annotate all failures with Explainer and with Baseline.  Retrain agent and evaluate'
# Original Training set with original human annotation, plus failures annotated by Explainer vs with Baseline
export SPLITS=$DATA_ROOT/splits/data_augmentation_experiment1_20200826.json 
# export SPLITS=$DATA_ROOT/splits/agent_augmentation_20200825.json # TOY
# export SPLITS=$DATA_ROOT/splits/debug_20200827.json # Debug
echo Split file $SPLITS

echo '-------------------------------'
# 1.1 With failure annotation by the explainer
export AUGMENTATION_LANG_MODEL=explainer
# training and eval_script
bash $SCRIPT_HOME/exp_1_train_eval.sh  # remove --fast_epoch add augmentation back

echo '-------------------------------'
# 1.2 With failure annotation by the baseline
export AUGMENTATION_LANG_MODEL=baseline
bash $SCRIPT_HOME/exp_1_train_eval.sh



# echo '--------------------------------------------------------------------------------------'
# echo 'Start Experiment 2 - Retrain agent on only half the original training data and evaluate '
# # Half of original training set
# export SPLITS=$ALFRED_ROOT/data/splits/data_augmentation_experiment2_20200831.json
# echo Split file $SPLITS

# echo '-------------------------------'
# # 2 Half of original training set, explainer/baseline are not used
# bash $SCRIPT_HOME/exp_2_train_eval.sh


# TODO 
# echo '--------------------------------------------------------------------------------------'
# echo 'Start Experiment 3 - Label the other half of the training data with the Explainer, retrain agent, eval '
# # Half of original training set, plus the other hald annotated by Explainer vs with Baseline
# export SPLITS=$DATA_ROOT/splits/?????????????????????????????????????????????????????????????
# echo Split file $SPLITS

# echo '-------------------------------'
# # 3.1 Half of original training set, plus the other hald annotated by Explainer
# export AUGMENTATION_LANG_MODEL=explainer
# # training and eval_script
# bash $SCRIPT_HOME/exp_3_train_eval.sh

# echo '-------------------------------'
# # 3.2 Half of original training set, plus the other hald annotated by Baseline
# export AUGMENTATION_LANG_MODEL=baseline
# # training and eval_script
# bash $SCRIPT_HOME/exp_3_train_eval.sh