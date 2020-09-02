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
echo 'Evaluate Experiment 1 - Annotate all failures with Explainer and with Baseline.  Eval on this Augmented Agent.'
# Original Training set with original human annotation, plus failures annotated by Explainer vs with Baseline
# export SPLITS=$DATA_ROOT/splits/data_augmentation_experiment1_20200826.json 
# export SPLITS=$DATA_ROOT/splits/agent_augmentation_20200825.json # TOY
# export SPLITS=$DATA_ROOT/splits/debug_20200827.json # Debug
# echo Split file $SPLITS

echo '-------------------------------'
# 1.1 With failure annotation by the explainer
export AUGMENTATION_LANG_MODEL=explainer
# training and eval_script
bash $SCRIPT_HOME/exp_1_eval.sh  # remove --fast_epoch add augmentation back

echo '-------------------------------'
# 1.2 With failure annotation by the baseline
# export AUGMENTATION_LANG_MODEL=baseline
# bash $SCRIPT_HOME/exp_1_eval.sh
