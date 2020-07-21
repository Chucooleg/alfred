# export SPLITS_FILE=$(ls $SPLITS_ROOT | grep -i 'demo_*' | tail -n 1)
# export SPLITS=$SPLITS_ROOT/$SPLITS_FILE

# cd $ALFRED_ROOT
# if [ $USE_GPU == 'gpu' ] ; then
#     echo 'Using gpu'
#   python models/run_demo/explain_fast_demo_trajectories.py --data $DATA --dout $EVAL_DATA_ROOT --splits $SPLITS --low_level_explainer_checkpt_path $EXPLAINER --high_level_explainer_checkpt_path $GOAL_EXPLAINER --gpu
# else
#     echo 'Not using gpu'
#   python models/run_demo/explain_fast_demo_trajectories.py --data $DATA --dout $EVAL_DATA_ROOT --splits $SPLITS --low_level_explainer_checkpt_path $EXPLAINER --high_level_explainer_checkpt_path $GOAL_EXPLAINER
# fi

# export EVAL_DATA_FILE=$(ls $EVAL_DATA_ROOT | grep -i 'new_trajectories*' | tail -n 1)
# export EVAL_DATA=$EVAL_DATA_ROOT/$EVAL_DATA_FILE
# printf $EVAL_DATA
# printf $SPLITS

# cd $ALFRED_ROOT/../alfred_human_eval
# python interface.py --split $SPLITS --data $EVAL_DATA --window_size 800

import os
import json
import pickle
import random
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import time
from datetime import datetime
from functools import partial

import re
import pprint

import sys
sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'models'))

import tkinter as tk
from tkinter import ttk
from tkinter import Canvas, StringVar, OptionMenu, Label, font
from tkinter import messagebox
import subprocess

class GameInterface():
    skill_map = {
        'Random Choice':'Random Choice', 
        'look_at_obj_in_light':'Examine item in light', 
        'pick_heat_then_place_in_recep':'Find, heat up and place item', 
        'pick_clean_then_place_in_recep':'Find, clean and place item',
        'pick_and_place_with_movable_recep': 'Find and place item with a container',
        'pick_and_place_simple': 'Find and place item',
        'pick_two_obj_and_place': 'Find and place two items',
        'pick_cool_then_place_in_recep': 'Find, cool and place item'
        }
    box_size = 20
    font_size = 20
    pad_size = 10

    def __init__(self, save, args):
        self.args = args
        self.save = save
        self.choices = {}

    def main_app(self):

        def skill_callback(eventObject):
            abc = eventObject.widget.get()
            skill_str = self.entries['skill'].get()
            self.choices['skill_int'] = skill_str.split('. ')[0]
            if self.choices['skill_int'] == '0':
                self.choices['skill_int'] = str(random.randint(1, 7))
            self.scene_type_idxs = sorted([int(idx) for idx in self.save['scene_type_based_lookup'][self.choices['skill_int']].keys()] + [0])
            scene_type_list = ['{}. {}'.format(idx, self.save["scenetype_set"][idx]) for idx in self.scene_type_idxs]
            self.entries['sceneType'].config(values=scene_type_list)

        def launch_game():

            # launch new game window
            last_traj_dir = sorted(os.listdir(self.args.eval_data_root))[-1]
            eval_data = os.path.join(self.args.eval_data_root, last_traj_dir)
            assert os.path.exists(eval_data)

            last_split_dirs = sorted(os.listdir(self.args.splits))
            last_split_dir = last_split_dirs[-2]
            split_file = os.path.join(self.args.splits, last_split_dir)
            assert os.path.exists(split_file)

            # call subprocess
            print('eval_data', eval_data)
            game_call_str = f'python ../alfred_human_eval/interface.py --split {split_file} --data {eval_data} --window_size 800'
            # game_call_str = 'pwd'
            subprocess.run(game_call_str, shell=True, check=True)

            self.game_root.quit()
            pass

        self.game_root = tk.Tk()
        self.entries = {}
        self.game_root.title('Skill Learning Tool For Game Players')

        # -------------------
        skills_list = ['{}. {}'.format(skill_i, self.skill_map[self.save["skill_set"][skill_i]]) for skill_i in sorted(self.save["skill_set"])]
        skill_row = tk.Frame(self.game_root)
        skill_lab = tk.Label(skill_row, text='I want to learn', anchor='w', font=("Helvetica", self.font_size))
        skill_ent = ttk.Combobox(skill_row, width=50, value=skills_list, font=("Helvetica", self.box_size))

        skill_row.pack(side=tk.TOP, fill=tk.X, padx=self.pad_size, pady=self.pad_size)
        skill_lab.pack(side=tk.LEFT)
        skill_ent.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.X)
        self.entries['skill'] = skill_ent
        # ---------------------------------------------------------
        scene_typ_row = tk.Frame(self.game_root)
        scene_typ_lab = tk.Label(scene_typ_row, text='Explore in this environment', anchor='w', font=("Helvetica", self.font_size))
        scene_typ_ent = ttk.Combobox(scene_typ_row, width=50, font=("Helvetica", self.box_size))

        scene_typ_row.pack(side=tk.TOP, fill=tk.X, padx=self.pad_size, pady=self.pad_size)
        scene_typ_lab.pack(side=tk.LEFT)
        scene_typ_ent.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.X)
        self.entries['sceneType'] = scene_typ_ent
        self.entries['sceneType'].bind('<Button-1>', skill_callback)
        # ---------------------------------------------------------        
        # TODO how did data arrive here?
        game_button_row = tk.Frame(self.game_root)
        self.game_button = tk.Button(game_button_row, text='Play Game', command=launch_game, font=("Helvetica", self.font_size))
        game_button_row.pack(side=tk.TOP, fill=tk.X, padx=self.pad_size, pady=self.pad_size)
        self.game_button.pack(side=tk.LEFT, padx=self.pad_size, pady=self.pad_size)
        # ---------------------------------------------------------  
        bigfont = font.Font(family="Helvetica",size=self.font_size)
        self.game_root.option_add("*TCombobox*Listbox*Font", bigfont)
        # -------------------
        self.game_root.mainloop()

    def run(self):
        self.main_app()

def main(args):
    save = pickle.load( open( "demo/task_lookup.p", "rb" ) )
    game_interface = GameInterface(save, args)
    game_interface.run()

if __name__ == '__main__':
 
    parser = ArgumentParser()
    parser.add_argument('--data', help='dataset folder', default='/data_alfred/json_demo_cache/')
    parser.add_argument('--dout', help='dataset folder', default='/data_alfred/demo_generated/')
    parser.add_argument('--splits', help='dataset folder', default='/data_alfred/splits/')   
    parser.add_argument('--eval_data_root', help='dataset folder', default='/data_alfred/demo_generated/')

    args = parser.parse_args()
    main(args)

