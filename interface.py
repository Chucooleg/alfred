import os
import cv2
import random
import json
import argparse
import copy
import numpy as np
import threading
import tkinter as tk
from tkinter import messagebox
import subprocess

from env.thor_env import ThorEnv

class Interface(threading.Thread):
    nid = 0
    title = ""
    message = ""

    def __init__(self, nid, title):
        threading.Thread.__init__(self)
        self.nid = nid
        self.title = title

        self.env = None
        self.reset()

    def display_note_gui(self):
        '''Tkinter to create a note gui window with parameters '''
        self.window = tk.Tk()
        self.window.title(self.title)
        self.window.geometry("200x400")
        self.setup_interface()
        self.window.mainloop()

    def reset(self):
        self.actions = list()
        self.select_object = ''
        self.fails = 0

    def set_env(self, env):
        self.env = env

    def step(self, action):
        try:
            event = self.env.step(action, smooth_nav=True)
            if not event.metadata['lastActionSuccess']:
                self.fails += 1
                print("Failed: " + str(event.metadata['errorMessage']))
            else:
                self.actions.append(action)
        except Exception as err:
            self.fails += 1
            print("Failed: %s" % (str(err)))

    def pickup(self):
        if self.select_object:
            action = dict(action="PickupObject",
                          objectId=self.select_object,
                          forceAction=True)
            self.step(action)

    def put(self):
        if self.select_object:
            if len(self.env.last_event.metadata['inventoryObjects']) > 0:
                inventory_object_id = self.env.last_event.metadata['inventoryObjects'][0]['objectId']
                action = dict(action="PutObject",
                              objectId=inventory_object_id,
                              receptacleObjectId=self.select_object,
                              forceAction=True,
                              placeStationary=True)
                self.step(action)
            else:
                print("Nothing in hand to put!")

    def open(self):
        if self.select_object:
            action = dict(action="OpenObject",
                          objectId=self.select_object,
                          moveMagnitude=1.0)
            self.step(action)

    def close(self):
        if self.select_object:
            action = dict(action="CloseObject",
                          objectId=self.select_object,
                          moveMagnitude=1.0)
            self.step(action)

    def toggleOn(self):
        if self.select_object:
            action = dict(action="ToggleObjectOn",
                          objectId=self.select_object)
            self.step(action)

    def toggleOff(self):
        if self.select_object:
            action = dict(action="ToggleObjectOff",
                          objectId=self.select_object)
            self.step(action)

    def slice(self):
        if self.select_object:
            inventory_objects = self.env.last_event.metadata['inventoryObjects']
            if len(inventory_objects) > 0 and 'Knife' in inventory_objects[0]['objectType']:
                action = dict(action='SliceObject',
                              objectId=self.select_object)
                self.step(action)

    def setup_interface(self):

        slice_btn = tk.Button(self.window, text='Slice', command=self.slice)
        slice_btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")

        toggle_off_btn = tk.Button(self.window, text='Toggle Off', command=self.toggleOff)
        toggle_off_btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")

        toggle_on_btn = tk.Button(self.window, text='Toggle On', command=self.toggleOn)
        toggle_on_btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")

        close_btn = tk.Button(self.window, text='Close', command=self.close)
        close_btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")

        open_btn = tk.Button(self.window, text='Open', command=self.open)
        open_btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")

        put_btn = tk.Button(self.window, text='Put', command=self.put)
        put_btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")

        pickup_btn = tk.Button(self.window, text='Pickup', command=self.pickup)
        pickup_btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")

    def run(self):
        self.display_note_gui()


class InstrDisplay(threading.Thread):
    nid = 0
    title = ""
    message = ""

    def __init__(self, nid, title):
        threading.Thread.__init__(self)
        self.nid = nid
        self.title = title

    def display_note_gui(self):
        '''Tkinter to create a note gui window with parameters '''
        print('InstrDisplay running display note gui')
        self.window = tk.Tk()
        self.window.title(self.title)
        self.window.geometry("700x600")
        self.setup_text_disp()
        self.window.mainloop()

    def setup_text_disp(self):
        self.txt = tk.Text(self.window, height=380, width=780)
        self.txt.pack()

    def run(self):
        self.display_note_gui()


class HumanEval(object):

    def __init__(self, args):
        self.args = args

        self.interface = Interface(0, "Select")
        self.interface.start()

        self.instr_display = InstrDisplay(1, "Instructions")
        self.instr_display.start()

        self.interface.select_object = ""
        self.frame_window_name = "Select"

        self.successes, self.failures = [], []
        self.actions = []
        self.results = {}

        with open(self.args.splits) as f:
            self.splits = json.load(f)

        random.seed(self.args.seed)


    # def get_tasks(self):
    #     seen_files, unseen_files = self.splits['tests_seen'], self.splits['tests_unseen']
    #     sample_size = int(len(seen_files) * self.args.fraction)

    #     seen_files = random.sample(seen_files, sample_size)
    #     unseen_files = random.sample(unseen_files, sample_size)

    #     for f in seen_files:
    #         f.update({'split': 'tests_seen'})
    #     for f in unseen_files:
    #         f.update({'split': 'tests_unseen'})

    #     return seen_files, unseen_files

    def get_tasks(self):

        k = list(self.splits.keys())[0]
        demo_files = self.splits[k]
        sample_size = max(1, int(len(demo_files) * self.args.fraction))

        demo_files = random.sample(demo_files, sample_size)

        for f in demo_files:
            f.update({'split': 'demo'})

        return demo_files

    @classmethod
    def setup_scene(cls, env, traj_data, args, reward_type='dense'):
        '''
        intialize the scene and agent from the task info
        '''
        # scene setup
        scene_num = traj_data['scene']['scene_num']
        object_poses = traj_data['scene']['object_poses']
        dirty_and_empty = traj_data['scene']['dirty_and_empty']
        object_toggles = traj_data['scene']['object_toggles']

        scene_name = 'FloorPlan%d' % scene_num
        env.reset(scene_name)
        env.restore_scene(object_poses, object_toggles, dirty_and_empty)

        # initialize to start position
        env.step(dict(traj_data['scene']['init_action']))

        # setup task for reward
        env.set_task(traj_data, args, reward_type=reward_type)


    def mouse_click_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            instance_segs = np.array(self.env.last_event.instance_segmentation_frame)
            color_to_object_id = self.env.last_event.color_to_object_id
            instance_color = tuple(instance_segs[y, x])
            obj = color_to_object_id[instance_color]
            if len([o for o in self.env.last_event.metadata['objects'] if o['objectId'] == obj]) > 0:
                self.interface.select_object = color_to_object_id[instance_color]
                print("Selected: " + self.interface.select_object)

                # update display
                img = np.uint8(self.env.last_event.frame)[:, :, ::-1].copy()
                mask = copy.copy(instance_segs)
                mask[:, :, :] = instance_segs == instance_color
                mask *= 255
                mask = np.uint8(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY))

                _, thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(img, contours, 0, (0, 0, 255), 2)
                cv2.imshow(self.frame_window_name, img)


    def eval_task(self, task):

        # json_path = os.path.join(self.args.data, task['split'], task['task'], 'traj_data.json')
        json_path = os.path.join(self.args.data, task['task'], 'traj_data.json')
        with open(json_path) as f:
            traj_data = json.load(f)

        self.setup_scene(self.env, traj_data, self.args)

        # goal instr
        ann = traj_data['explainer_annotations']['anns'][0]
        goal_instr = None if ann['task_desc'] == '' else ann['task_desc']
        sbs_instrs = ann['high_descs']
        
        if goal_instr is not None:
            instr_str = "GOAL:\n%s\n\nINSTRUCTIONS:\n%s" % (goal_instr, '\n'.join(sbs_instrs))
        else:
            instr_str = "INSTRUCTIONS:\n%s" % ('\n'.join(sbs_instrs))

        self.instr_display.txt.delete('1.0', tk.END)
        self.instr_display.txt.insert(tk.END, instr_str)

        # image
        self.update_frame_window()
        # mouse callback
        cv2.setMouseCallback('Select', self.mouse_click_cb)
        # user input on screen
        k = cv2.waitKey()

        t = 0
        reward = 0
        done, success = False, False
        self.interface.reset()

        while not done:
            if t > args.max_steps:
                print("Exceeded max steps (%d). Task Failed" % args.max_steps)
                break

            if k == 27:
                break
            elif k == ord('w'):
                action = {'action': 'MoveAhead'}
                self.interface.step(action)
            elif k == ord('s'):
                action = {'action': 'MoveBack'}
                self.interface.step(action)
            elif k == ord('a'):
                action = {'action': 'RotateLeft'}
                self.interface.step(action)
            elif k == ord('d'):
                action = {'action': 'RotateRight'}
                self.interface.step(action)
            elif k == ord('='):
                action = {'action': 'LookUp'}
                self.interface.step(action)
            elif k == ord('-'):
                action = {'action': 'LookDown'}
                self.interface.step(action)
            elif k == ord('`'):
                self.update_frame_window()
                k = cv2.waitKey()
                continue

            elif k == ord('r'):
                resp = tk.messagebox.askquestion('Task', 'Reset task?', icon='warning')
                if resp == "yes":
                    t = 0
                    reward = 0
                    done, success = False, False
                    self.interface.reset()
                    self.setup_scene(self.env, traj_data, self.args)

            elif k == ord('n'):
                resp = tk.messagebox.askquestion('Task', 'Done? Next Task?', icon='warning')
                if resp == "yes":
                    break

            if self.interface.fails > args.max_fails:
                print("Too many failures (>%d). Task Failed" % args.max_fails)
                resp = tk.messagebox.askquestion('Task', 'Too many failures. Retry?', icon='warning')
                if resp == 'yes':
                    t = 0
                    reward = 0
                    done, success = False, False
                    self.interface.reset()
                    self.setup_scene(self.env, traj_data, self.args)
                else:
                    break

            t_reward, t_done = self.env.get_transition_reward()
            reward += t_reward
            t += 1

            self.update_frame_window()
            k = cv2.waitKey()
            self.interface.select_object = ''


        # check if goal was satisfied
        goal_satisfied = self.env.get_goal_satisfied()
        if goal_satisfied:
            print("Goal Reached")
            success = True


        # postconditions
        pcs = self.env.get_postconditions_met()
        postcondition_success_rate = pcs[0] / float(pcs[1])

        # SPL
        path_len_weight = len(traj_data['plan']['low_actions'])
        s_spl = (1 if goal_satisfied else 0) * min(1., path_len_weight / float(t))
        pc_spl = postcondition_success_rate * min(1., path_len_weight / float(t))

        # path length weighted SPL
        plw_s_spl = s_spl * path_len_weight
        plw_pc_spl = pc_spl * path_len_weight

        # log success/fails
        log_entry = {'trial': traj_data['task_id'],
                     'type': traj_data['task_type'],
                     'repeat_idx': int(r_idx),
                     'goal_instr': goal_instr,
                     'completed_postconditions': int(pcs[0]),
                     'total_postconditions': int(pcs[1]),
                     'postcondition_success': float(postcondition_success_rate),
                     'success_spl': float(s_spl),
                     'path_weighted_success_spl': float(plw_s_spl),
                     'postcondition_spl': float(pc_spl),
                     'path_weighted_postcondition_spl': float(plw_pc_spl),
                     'path_len_weight': int(path_len_weight),
                     'reward': float(reward)}

        # success and failures
        if success:
            self.successes.append(log_entry)
        else:
            self.failures.append(log_entry)


        # actions
        actseq = {traj_data['task_id']: list(self.interface.actions)}
        self.actions.append(actseq)

        # stats
        num_successes, num_failures = len(self.successes), len(self.failures)
        num_evals = len(self.successes) + len(self.failures)
        total_path_len_weight = sum([entry['path_len_weight'] for entry in self.successes]) + \
                                sum([entry['path_len_weight'] for entry in self.failures])
        completed_postconditions = sum([entry['completed_postconditions'] for entry in self.successes]) + \
                                   sum([entry['completed_postconditions'] for entry in self.failures])
        total_postconditions = sum([entry['total_postconditions'] for entry in self.successes]) + \
                               sum([entry['total_postconditions'] for entry in self.failures])

        # metrics
        sr = float(num_successes) / num_evals
        pc = completed_postconditions / float(total_postconditions)
        plw_sr = (float(sum([entry['path_weighted_success_spl'] for entry in self.successes]) +
                                    sum([entry['path_weighted_success_spl'] for entry in self.failures])) /
                                    total_path_len_weight)
        plw_pc = (float(sum([entry['path_weighted_postcondition_spl'] for entry in self.successes]) +
                                    sum([entry['path_weighted_postcondition_spl'] for entry in self.failures])) /
                                    total_path_len_weight)


        # save results
        self.results['success'] = {'num_successes': num_successes,
                              'num_evals': num_evals,
                              'success_rate': sr}
        self.results['postcondition_success'] = {'completed_postconditions': completed_postconditions,
                                            'total_postconditions': total_postconditions,
                                            'postcondition_success_rate': pc}
        self.results['path_length_weighted_success_rate'] = plw_sr
        self.results['path_length_weighted_postcondition_success_rate'] = plw_pc

        print("-------------")
        print("SR: %d/%d = %.3f" % (num_successes, num_evals, sr))
        print("PC: %d/%d = %.3f" % (completed_postconditions, total_postconditions, pc))
        print("PLW S: %.3f" % (plw_sr))
        print("PLW PC: %.3f" % (plw_pc))
        print("-------------")



    def update_frame_window(self):
        img = np.uint8(self.env.last_event.frame)[:, :, ::-1]
        cv2.imshow(self.frame_window_name, img)


    def save_results(self):
        results = {'successes': list(self.successes),
                   'failures': list(self.failures),
                   'results': dict(self.results),
                   'actions': list(self.actions)}

        save_path = os.path.join('results', args.results_file)
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=4, sort_keys=True)

    def load_results(self):
        json_path = os.path.join('results', args.results_file)
        with open(json_path, 'r') as f:
            res = json.load(f)

        self.successes = res['successes']
        self.failures = res['failures']
        self.results = res['results']
        self.actions = res['actions']
        return len(self.successes) + len(self.failures)

    def run(self):

        def run_eval(t):
            self.eval_task(t)
            self.save_results()

        self.env = ThorEnv(player_screen_height=self.args.window_size,
                      player_screen_width=self.args.window_size)
        self.interface.set_env(self.env)

        if args.resume:
            num_completed_tasks = self.load_results()
        else:
            num_completed_tasks = 0

        demo_files = self.get_tasks()

        start, end =  args.start_idx, args.end_idx
        # for t in seen_files:
        #     run_eval(t)
        for i, t in enumerate(demo_files[start+num_completed_tasks:end]):
            print("(+%d) %d/%d Tasks Finished." % (start, i+num_completed_tasks, end-start))
            run_eval(t)

        print("All Tasks Completed. Thanks you for your help!")

        self.env.stop()
        cv2.destroyAllWindows()

        self.interface.join()
        self.instr_display.join()

if __name__ == '__main__':
    # parser
    parser = argparse.ArgumentParser()

    # settings
    parser.add_argument('--splits', type=str, default='data/splits/oct21.json')
    parser.add_argument('--data', type=str, default='data/secret_2.1.0')
	# ???
    parser.add_argument('--results_file', type=str, default='human_eval_results_2020.json')
    parser.add_argument('--fraction', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--reward_config', default='models/config/rewards.json')
    parser.add_argument('--max_steps', type=int, default=1000)
    parser.add_argument('--max_fails', type=int, default=10)
    parser.add_argument('--max_resets', type=int, default=100)
    parser.add_argument('--window_size', type=int, default=300)
    parser.add_argument('--resume', dest='resume', action='store_true')
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=100)

    # parse arguments
    args = parser.parse_args()
    args.reward_config = os.path.join(os.environ['ALFRED_ROOT'], args.reward_config)

    subprocess.call(["pkill", "-f", 'thor'])

    # human eval
    human_eval = HumanEval(args)
    human_eval.run()
