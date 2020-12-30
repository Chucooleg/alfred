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



echo '--------------------------------------------------------------------------------------'
echo 'Start Experiment 1 - Annotate all failures with Explainer Full, Explainer Aux Loss Only, Baseline'
# Original Training set with original human annotation, plus new 12k annotated by Explainer full, Explainer Aux Loss only, and Baseline
export SPLITS=$DATA_ROOT/splits/data_augmentation_experiment1_20201230.json 
# export SPLITS=$DATA_ROOT/splits/agent_augmentation_20201230_toy.json # TOY
echo Split file $SPLITS

echo '-------------------------------'
# 1.1 With failure annotation by the explainer
export AUGMENTATION_LANG_MODEL=explainer_full
# training and eval_script
bash $SCRIPT_HOME/exp_1_train_azvm.sh