import os
import json
import pickle
import random
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import time
from datetime import datetime
from functools import partial

import threading
import tkinter as tk
from tkinter import ttk
from tkinter import Canvas, StringVar, OptionMenu, Label, font
from tkinter import messagebox
import subprocess


# class DevInterface(threading.Thread):
class DevInterface():
    fields = ['skill_int', 'pickupObject_int', 'movable_int', 'receptacle_int', 'scene_int']
    box_size = 15
    font_size = 15

    def __init__(self, save, args):
        self.save = save
        self.args = args
        self.choices = {}

    # (Pdb) new_split_path
    # '/data_alfred/splits/demo_T20200704_141212_436106.json'
    # (Pdb) new_split
    # {'valid_unseen': [{'task': 'pick_two_obj_and_place-AppleSliced-None-CounterTop-10/trial_T20190907_061347_004735', 'repeat_idx': 0}]}

    def retrieve_human_annotation(self, new_split_path, new_split):
        tar_split = list(new_split.keys())[0]
        raw_traj_path = os.path.join(self.args.data, tar_split, new_split[tar_split][0]['task'], 'traj_data.json')
        with open(raw_traj_path, 'r') as f:
            ann = json.load(f)['turk_annotations']['anns'][0]
        ann_formatted = ann['task_desc'] + '\n\n' + '\n'.join(ann['high_descs'])        
        return ann_formatted

    def run_baseline_prediction(self, new_split_path, new_split):
        pass

    def run_explainer(self, new_split_path, new_split):
        pass

    def main_app(self):
        self.query_root = tk.Tk()
        self.entries = {}
        self.query_root.title('Skill Teaching Tool')

        def skill_callback(eventObject):
            abc = eventObject.widget.get()
            skill_str = self.entries['skill'].get()
            self.choices['skill_int'] = skill_str.split('. ')[0]
            if self.choices['skill_int'] == '0':
                self.choices['skill_int'] = str(random.randint(1, 7)) 
            self.pickup_object_idxs = sorted([int(idx) for idx in self.save['lookup'][self.choices['skill_int']].keys()])
            pickup_object_list = ['{}. {}'.format(idx, self.save["pickupObject_set"][idx]) for idx in self.pickup_object_idxs]
            self.entries['pickupObject'].config(values=pickup_object_list) 

        def pickup_callback(eventObject):
            abc = eventObject.widget.get()
            pickup_str = self.entries['pickupObject'].get()
            self.choices['pickupObject_int'] = pickup_str.split('. ')[0]
            if self.choices['pickupObject_int'] == '0':
                self.choices['pickupObject_int'] = str(random.choice(self.pickup_object_idxs))
            if self.save['skill_set'][int(self.choices['skill_int'])] == 'pick_and_place_with_movable_recep':
                self.movable_idxs = sorted([int(idx) for idx in self.save['lookup'][self.choices['skill_int']][self.choices['pickupObject_int']].keys()])
                movable_list = ['{}. {}'.format(idx, self.save["movable_set"][idx]) for idx in self.movable_idxs]
            else:
                self.movable_idxs = [1]
                movable_list = ['1. None']
            self.entries['movable'].config(values=movable_list)

        def movable_callback(eventObject):
            abc = eventObject.widget.get()
            movable_str = self.entries['movable'].get()
            self.choices['movable_int'] = movable_str.split('. ')[0]
            if self.choices['movable_int'] == '0':
                self.choices['movable_int'] = str(random.choice(self.movable_idxs))
            self.receptacle_idxs = sorted([int(idx) for idx in self.save['lookup'][self.choices['skill_int']][self.choices['pickupObject_int']][self.choices['movable_int']].keys()])
            receptacle_list = ['{}. {}'.format(idx, self.save["receptacle_set"][idx]) for idx in self.receptacle_idxs]
            self.entries['receptacle'].config(values=receptacle_list)

        def generate_traj_callback():
            receptacle_str = self.entries['receptacle'].get()
            self.choices['receptacle_int'] = receptacle_str.split('. ')[0]
            if self.choices['receptacle_int'] == '0':
                self.choices['receptacle_int'] = str(random.choice(self.receptacle_idxs))
            self.scene_idxs = [int(idx) for idx in self.save['lookup'][self.choices['skill_int']][self.choices['pickupObject_int']][self.choices['movable_int']][self.choices['receptacle_int']].keys()]
            self.choices['scene_int'] = random.choice(self.scene_idxs)
            self.task_def = (
                self.save['skill_set'][int(self.choices['skill_int'])], 
                self.save['pickupObject_set'][int(self.choices['pickupObject_int'])], 
                self.save['movable_set'][int(self.choices['movable_int'])], 
                self.save['receptacle_set'][int(self.choices['receptacle_int'])], 
                self.choices['scene_int'])
            
            # TODO make instructions
            new_split_path, new_split = make_new_split()
            human_annotation = self.retrieve_human_annotation(new_split_path, new_split)
            baseline_prediction = self.run_baseline_prediction(new_split_path, new_split)
            explainer_prediction = self.run_explainer(new_split_path, new_split)

            # update instructions in window 
            goal_str = '-----------------Goal Directed Instructions-----------------'
            base_str = '-----------------Baseline Instructions----------------------'
            human_str = '-----------------Human Annotation---------------------------'
            self.task_text.set(f"{goal_str}\n{explainer_prediction}\n\n\n{base_str}\n{baseline_prediction}\n\n\n{human_str}\n{human_annotation}\n")

            # TODO add model, add instructions
            # https://github.com/Chucooleg/alfred/blob/60c89e0c5d0d16e1b8f5b5a562a23dc89d4f7bb5/demo/query_and_make_split.py

        def make_new_split():
            # get trajectory ID
            splits_and_trajs = self.save['lookup'][str(self.choices['skill_int'])][str(self.choices['pickupObject_int'])][str(self.choices['movable_int'])][str(self.choices['receptacle_int'])][str(self.choices['scene_int'])]
            split = random.choice(list(splits_and_trajs.keys()))
            self.choices['traj_id'] = random.choice(splits_and_trajs[split])
            # make the new split file
            task_name = '{}-{}-{}-{}-{}'.format(self.save["skill_set"][int(self.choices['skill_int'])], self.save["pickupObject_set"][int(self.choices['pickupObject_int'])], self.save["movable_set"][int(self.choices['movable_int'])], self.save["receptacle_set"][int(self.choices['receptacle_int'])], str(self.choices['scene_int']))

            new_split = {split: [{'task': f'{task_name}/{self.choices["traj_id"]}', 'repeat_idx':0}]}
            TIME_NOW = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            new_split_path = os.path.join(self.args.splits, f'demo_T{TIME_NOW}.json')
            with open(new_split_path, 'w') as f:
                json.dump(new_split, f)
            return new_split_path, new_split  

        def generate_game_preview():
            # launch new game window
            # TODO
            # call new process?
            pass


        # -------------------
        skills_list = ['{}. {}'.format(skill_i, self.save["skill_set"][skill_i]) for skill_i in sorted(self.save["skill_set"])]
        skill_row = tk.Frame(self.query_root)
        skill_lab = tk.Label(skill_row, text='Choose a skill to learn', anchor='w', font=("Helvetica", self.font_size))
        skill_ent = ttk.Combobox(skill_row, width=50, value=skills_list, font=("Helvetica", self.box_size))

        skill_row.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        skill_lab.pack(side=tk.LEFT)
        skill_ent.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.X)
        self.entries['skill'] = skill_ent
        # -------------------
        pickup_object_row = tk.Frame(self.query_root)
        pickup_object_lab = tk.Label(pickup_object_row, text='Choose an object to pickup', anchor='w', font=("Helvetica", self.font_size))
        pickup_object_ent = ttk.Combobox(pickup_object_row, width=50, font=("Helvetica", self.box_size))

        pickup_object_row.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        pickup_object_lab.pack(side=tk.LEFT)
        pickup_object_ent.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.X)
        self.entries['pickupObject'] = pickup_object_ent
        self.entries['pickupObject'].bind('<Button-1>', skill_callback)
        # -------------------
        movable_row = tk.Frame(self.query_root)
        movable_lab = tk.Label(movable_row, text='Choose a container to place the object', anchor='w', font=("Helvetica", self.font_size))
        movable_ent = ttk.Combobox(movable_row, width=50, font=("Helvetica", self.box_size))

        movable_row.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        movable_lab.pack(side=tk.LEFT)
        movable_ent.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.X)
        self.entries['movable'] = movable_ent
        self.entries['movable'].bind('<Button-1>', pickup_callback)
        # -------------------
        receptacle_row = tk.Frame(self.query_root)
        receptacle_lab = tk.Label(receptacle_row, text='Choose a final place for the objects', anchor='w', font=("Helvetica", self.font_size))
        receptacle_ent = ttk.Combobox(receptacle_row, width=50, font=("Helvetica", self.box_size))

        receptacle_row.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        receptacle_lab.pack(side=tk.LEFT)
        receptacle_ent.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.X)
        self.entries['receptacle'] = receptacle_ent
        self.entries['receptacle'].bind('<Button-1>', movable_callback)
        # -------------------
        generate_button_row = tk.Frame(self.query_root)
        self.generate_button = tk.Button(generate_button_row, text='Generate Instructions', command=generate_traj_callback, font=("Helvetica", self.font_size))
        generate_button_row.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        self.generate_button.pack(side=tk.LEFT, padx=5, pady=5)

        description_row = tk.Frame(self.query_root)
        self.task_text = tk.StringVar()
        goal_str = '-----------------Goal Directed Instructions-----------------'
        base_str = '-----------------Baseline Instructions----------------------'
        human_str = '-----------------Human Annotation---------------------------'
        self.task_text.set(f"{goal_str}\n\n{base_str}\n\n{human_str}")
        self.task_label = tk.Label(description_row, textvariable=self.task_text, font=("Helvetica", self.font_size), justify='left')
        description_row.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        self.task_label.pack(side=tk.LEFT, padx=5, pady=5, fill='both')
        # -------------------
        preview_button_row = tk.Frame(self.query_root)
        self.preview_button = tk.Button(preview_button_row, text='Preview Game', command=generate_game_preview, font=("Helvetica", self.font_size))
        preview_button_row.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        self.preview_button.pack(side=tk.LEFT, padx=5, pady=5)

        # -------------------
        self.query_root.mainloop()

    def run(self):
        self.main_app()
       

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--data', help='dataset folder', default='/data_alfred/json_demo_cache/')
    parser.add_argument('--splits', help='dataset folder', default='/data_alfred/splits/')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    save = pickle.load( open( "demo/task_lookup.p", "rb" ) )
    dev_interface = DevInterface(save, args)
    dev_interface.run()
