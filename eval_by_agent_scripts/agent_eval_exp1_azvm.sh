# First, Run tmux into the container
# tmux attach -t container


# General Path Setup
export ENV_HOME=/root/home/hoyeung
export ALFRED_ROOT=$ENV_HOME/alfred
export DATA_ROOT=/root/data_alfred
export BLOB_ROOT=$ENV_HOME/blob_alfred_data/ # for backup to container
export BLOB_EXP_DIR=$BLOB_ROOT/exp_all/
export DATA=$DATA_ROOT/json_feat_2.1.0_backup_20200826_agent_training
export PP=pp
export MODEL=seq2seq_im_mask
export AUGMENTATION_DATA=$DATA_ROOT/unlabeled_12k_20201206/seen/

# where this script is
export SCRIPT_HOME=$ALFRED_ROOT/eval_by_agent_scripts/
cd $SCRIPT_HOME


echo '-------------------------------'
echo 'Start Eval Experiment 1'
# 1.1 With failure annotation by the explainer
export AUGMENTATION_LANG_MODEL=$1
export SEEN_EPOCH=$2
export UNSEEN_EPOCH=$3
bash $SCRIPT_HOME/exp_1_eval_azvm.sh