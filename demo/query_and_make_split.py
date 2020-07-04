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
    # nid = 0
    # title = ""
    # message = ""
    fields = ['skill_int', 'pickupObject_int', 'movable_int', 'receptacle_int', 'scene_int']
    box_size = 15
    font_size = 15

    def __init__(self, save):
        self.save = save
        self.choices = {}
    
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

        def receptacle_callback():
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
            import pdb; pdb.set_trace()

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
        self.generate_button = tk.Button(self.query_root, text='Generate', command=receptacle_callback, font=("Helvetica", self.font_size))
        self.generate_button.pack(side=tk.LEFT, padx=5, pady=5)
       
        self.query_root.mainloop()

    def run(self):
        self.main_app()
       
        
# https://www.python-course.eu/tkinter_entry_widgets.php

    #     threading.Thread.__init__(self)
    #     self.nid = nid
    #     self.choices = {'skill_int':None, 'pickupObject_int':None, 'movable_int':None, 'receptacle_int':None, 'scene_int':0}
    #     self.save=save
    #     # self.reset()

    # def display_note_gui(self):
    #     '''Create a note window'''

    #     def skill_callback(eventObject):
    #         abc = eventObject.widget.get()
    #         skill_str = self.skill_combo.get()
    #         self.choices['skill_int'] = skill_str.split('. ')[0]
    #         if self.choices['skill_int'] == '0':
    #             self.choices['skill_int'] = str(random.randint(1, 7)) 
    #         self.pickup_object_idxs = sorted([int(idx) for idx in self.save['lookup'][self.choices['skill_int']].keys()])
    #         pickup_object_list = ['{}. {}'.format(idx, self.save["pickupObject_set"][idx]) for idx in self.pickup_object_idxs]
    #         self.pickup_object_combo.config(values=pickup_object_list)      

    #     def pickup_callback(eventObject):
    #         abc = eventObject.widget.get()
    #         pickup_str = self.pickup_object_combo.get()
    #         self.choices['pickupObject_int'] = pickup_str.split('. ')[0]
    #         if self.choices['pickupObject_int'] == '0':
    #             self.choices['pickupObject_int'] = str(random.choice(self.pickup_object_idxs))
    #         if self.save['skill_set'][int(self.choices['skill_int'])] == 'pick_and_place_with_movable_recep':
    #             self.movable_idxs = sorted([int(idx) for idx in self.save['lookup'][self.choices['skill_int']][self.choices['pickupObject_int']].keys()])
    #             movable_list = ['{}. {}'.format(idx, self.save["movable_set"][idx]) for idx in self.movable_idxs]
    #         else:
    #             self.movable_idxs = [1]
    #             movable_list = ['1. None']
    #         self.movable_combo.config(values=movable_list)   

    #     def movable_callback(eventObject):
    #         abc = eventObject.widget.get()
    #         movable_str = self.movable_combo.get()
    #         self.choices['movable_int'] = movable_str.split('. ')[0]
    #         if self.choices['movable_int'] == '0':
    #             self.choices['movable_int'] = str(random.choice(self.movable_idxs))
    #         self.receptacle_idxs = sorted([int(idx) for idx in self.save['lookup'][self.choices['skill_int']][self.choices['pickupObject_int']][self.choices['movable_int']].keys()])
    #         # receptacle_idxs = sorted([int(idx) for idx in self.save['lookup'][self.choices['skill_int']][self.choices['pickupObject_int']].keys()])
    #         receptacle_list = ['{}. {}'.format(idx, self.save["receptacle_set"][idx]) for idx in self.receptacle_idxs]
    #         self.receptacle_combo.config(values=receptacle_list)

    #     def receptacle_callback():
    #         # abc = eventObject.widget.get()
    #         receptacle_str = self.receptacle_combo.get()
    #         self.choices['receptacle_int'] = receptacle_str.split('. ')[0]
    #         if self.choices['receptacle_int'] == '0':
    #             self.choices['receptacle_int'] = str(random.choice(self.receptacle_idxs))
    #         self.scene_idxs = [int(idx) for idx in self.save['lookup'][self.choices['skill_int']][self.choices['pickupObject_int']][self.choices['movable_int']][self.choices['receptacle_int']].keys()]
    #         self.choices['scene_int'] = random.choice(self.scene_idxs)
    #         self.task_text.set(f"<{self.save['skill_set'][int(self.choices['skill_int'])]} {self.save['pickupObject_set'][int(self.choices['pickupObject_int'])]}  {self.save['movable_set'][int(self.choices['movable_int'])]}  {self.save['receptacle_set'][int(self.choices['receptacle_int'])]}>")

    #     self.master_window = tk.Tk()
    #     self.master_window.title('Skill Teaching Tool')
    #     self.master_window.geometry("2000x400")

    #     self.skill_label = ttk.Label(self.master_window, text='Choose a skill to learn', font=("Helvetica", 15))
    #     self.skill_label.grid(column=0, row=1, sticky = 'E')

    #     skills_list = ['{}. {}'.format(skill_i, self.save["skill_set"][skill_i]) for skill_i in sorted(self.save["skill_set"])]
    #     self.skill_combo = ttk.Combobox(self.master_window, width=50, value=skills_list, font=("Helvetica", 15))
    #     # self.skill_combo.grid(column=1, row=1)
    #     self.skill_combo.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
    #     # row.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
    #     # 

    #     self.pickup_object_label = ttk.Label(self.master_window, text='Choose an object to pickup', font=("Helvetica", 15))
    #     self.pickup_object_label.grid(column=0, row=3, sticky = 'E')        

    #     self.pickup_object_combo = ttk.Combobox(self.master_window, width=50, font=("Helvetica", 15))
    #     self.pickup_object_combo.grid(column=1, row=3)
    #     self.pickup_object_combo.bind('<Button-1>', skill_callback)

    #     # 

    #     self.movable_label = ttk.Label(self.master_window, text='Choose a container to place the object', font=("Helvetica", 15))
    #     self.movable_label.grid(column=0, row=5, sticky = 'E')        

    #     self.movable_combo = ttk.Combobox(self.master_window, width=50, font=("Helvetica", 15))
    #     self.movable_combo.grid(column=1, row=5)
    #     self.movable_combo.bind('<Button-1>', pickup_callback)

    #     # 

    #     self.receptacle_label = ttk.Label(self.master_window, text='Choose a final place for the objects', font=("Helvetica", 15))
    #     self.receptacle_label.grid(column=0, row=7, sticky = 'E')        

    #     self.receptacle_combo = ttk.Combobox(self.master_window, width=50, font=("Helvetica", 15))
    #     self.receptacle_combo.grid(column=1, row=7)
    #     self.receptacle_combo.bind('<Button-1>', movable_callback)


    #     self.generate_button = tk.Button(self.master_window, text="Generate Task", command=receptacle_callback, width=49, font=("Helvetica", 15))
    #     self.generate_button.grid(column=1, row=9)

    #     self.task_text = tk.StringVar()
    #     self.task_text.set("<Task Definition>")
    #     self.task_label = Label(self.master_window, textvariable=self.task_text, font=("Helvetica", 15))
    #     self.task_label.grid(column=0, row=11)

    #     # put 
    #     # TODO https://www.python-course.eu/tkinter_entry_widgets.php

    #     # combo box font
    #     bigfont = font.Font(family="Helvetica",size=15)
    #     self.master_window.option_add("*TCombobox*Listbox*Font", bigfont)
    #     self.master_window.grid_columnconfigure(4, minsize=10000)





def run_app():
    save = pickle.load( open( "demo/task_lookup.p", "rb" ) )
    # import pdb; pdb.set_trace()
    dev_interface = DevInterface(save)
    dev_interface.run()

    # dev_interface.start()

    # dev_interface.join()

if __name__ == '__main__':
    run_app()

# from tkinter import *

# master = Tk()

# def do_something_():
#     print('do something') #I added this so that i can run the code with no errors
#     #*performing a function on widget*

# def resetAll():
#     canvas.destroy() #destroys the canvas and therefore all of its child-widgets too


# canvas = Canvas(master)
# canvas.pack()
# #creates the cnvas

# DoThing = Button(canvas, text='Do Something',command=do_something_).pack(pady=10) 
# #its master widget is now the canvas

# clearall = Button(canvas, text='reset', command=resetAll).pack(pady=10)
# #its master widget is now the canvas

# master.mainloop()




# if __name__ == "__main__":

#     parser = ArgumentParser()
#     parser.add_argument('--data', help='dataset folder', default='/root/data_alfred/json_feat_2.1.0/')
#     parser.add_argument('--splits', help='dataset folder', default='/root/data_alfred/splits/')
#     parser.add_argument('--debug', action='store_true')
#     args = parser.parse_args()

#     print('\n\n\nWelcome to the skill-learning demo.\n')

#     # 1. choose skill
#     skill_int = 999
#     while not skill_int in range(0,8):
#         print('\nPlease choose a skill:')
#         for i in range(0,8):
#             print (f'{i}. {save["skill_set"][i]}')
#         skill_int = input('\nSkill choice (Valid Number Only): ')
#         skill_int = int(skill_int)
#     if skill_int == 0:
#         skill_int = random.randint(1, 7) 
#     print (f'Skill = {save["skill_set"][skill_int]}')

#     # 2. choose pickup object
#     idxs = sorted([int(idx) for idx in save['lookup'][str(skill_int)].keys()])
#     pickupObject_int = 999
#     while not pickupObject_int in [0] + idxs:
#         print('\nPlease choose a pickup object:')
#         for idx in [0] + idxs:
#             print(f'{idx}. {save["pickupObject_set"][idx]}')
#         pickupObject_int = input('\nPickup Object choice (Valid Number Only): ')
#         pickupObject_int = int(pickupObject_int)
#     if pickupObject_int == 0:
#         pickupObject_int = random.choice(idxs)
#     print (f'Pickup Object = {save["pickupObject_set"][pickupObject_int]}')

#     # 3. choose movable receptacle
#     if save["skill_set"][skill_int] == 'pick_and_place_with_movable_recep':
#         idxs = sorted([int(idx) for idx in save['lookup'][str(skill_int)][str(pickupObject_int)].keys()])
#         movable_int = 999
#         while not movable_int in [0] + idxs:
#             print('\nPlease choose a movable receptacle:')
#             for idx in [0] + idxs:
#                 print(f'{idx}. {save["movable_set"][idx]}')
#             movable_int = input('\nMovable Receptacle choice (Valid Number Only): ')
#             movable_int = int(movable_int)
#         if movable_int == 0:
#             movable_int = random.choice(idxs)
#         print (f'Movable Receptacle = {save["movable_set"][movable_int]}')
#     else:
#         movable_int = save["movable2num"]['None']
#         print (f'Movable Receptacle = {save["movable_set"][movable_int]}')

#     # 4. choose a final receptacle
#     idxs = sorted([int(idx) for idx in save['lookup'][str(skill_int)][str(pickupObject_int)][str(movable_int)].keys()])
#     receptacle_int = 999
#     while not receptacle_int in [0] + idxs:
#         print('\nPlease choose a final receptacle:')
#         for idx in [0] + idxs:
#             print(f'{idx}. {save["receptacle_set"][idx]}')
#         receptacle_int = input('\nFinal Receptacle choice (Valid Number Only): ')
#         receptacle_int = int(receptacle_int)
#     if receptacle_int == 0:
#         receptacle_int = random.choice(idxs)
#     print (f'Final Receptacle = {save["receptacle_set"][receptacle_int]}')

#     # 5. choose a scene
#     idxs = sorted([int(idx) for idx in save['lookup'][str(skill_int)][str(pickupObject_int)][str(movable_int)][str(receptacle_int)].keys()])
#     print(f'\nAvailable scene numbers : {idxs}')
#     scene_num = random.choice(idxs)
#     print(f'Sampled random scene {scene_num}.')

#     # print tuple
#     task_tuple = (save["skill_set"][skill_int], save["pickupObject_set"][pickupObject_int], save["movable_set"][movable_int], save["receptacle_set"][receptacle_int], scene_num)
#     if args.debug:
#         print(f'\nTask tuple: {task_tuple}' )

#     # make split
#     # {'valid_seen': [T_..., T_...], 'valid_unseen': [T_..., T_...]}
#     splits_and_trajs = save['lookup'][str(skill_int)][str(pickupObject_int)][str(movable_int)][str(receptacle_int)][str(scene_num)]
#     split = random.choice(list(splits_and_trajs.keys()))
#     traj_id = random.choice(splits_and_trajs[split])
#     if args.debug:
#         print( split, traj_id)

#     # return a video link    
#     task_name = '{}-{}-{}-{}-{}'.format(save["skill_set"][skill_int], save["pickupObject_set"][pickupObject_int], save["movable_set"][movable_int], save["receptacle_set"][receptacle_int], scene_num)
#     url = f'https://mturk.jessethomason.com/lang_2_plan/2.1.0/{task_name}/{traj_id}/video.mp4'
#     print (f'\n\n\nClick to watch Planner Sampled Trajectory:\n {url}\n\n\n\n\n\n\n')

#     # make new split
#     new_split = {split: [{'task': f'{task_name}/{traj_id}', 'repeat_idx':0}]}
#     TIME_NOW = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
#     new_split_path = os.path.join(args.splits, f'demo_T{TIME_NOW}.json')
#     with open(new_split_path, 'w') as f:
#         json.dump(new_split, f)
#     if args.debug:
#         print('Made new split file',new_split_path)
#     time.sleep(2)
#     print('-------------------------------------------------------------------------------')


