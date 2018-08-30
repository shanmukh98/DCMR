import mujoco_py
import numpy as np
from pyquaternion import Quaternion

# class robot:
#     def
# class bot:
#     def __init__(self.id):
#         self.position = 0
#         self.orientation = 0
#         self.view_front = []
#         self.view_back = []
#         self.

class env:
    def __init__(self, path_to_xml, sim_time_step, ctrl_time_step, episode_time):
        self.path_to_xml = path_to_xml
        self.model = mujoco_py.load_model_from_path(path_to_xml)
        self.sim = mujoco_py.MjSim(self.model)
        self.sim_time_step = sim_time_step
        self.ctrl_time_step = ctrl_time_step
        self.view_front = []
        self.view_back = []
        self.qpos_1 = self.sim.data.get_joint_qpos("bot_1")
        self.qpos_2 = self.sim.data.get_joint_qpos("bot_2")
        self.qvel_1 = self.sim.data.get_joint_qvel("bot_1")
        self.qvel_2 = self.sim.data.get_joint_qvel("bot_2")
        self.time = 0
        self.episode_time = episode_time
        self.episode_state = 0
        self.joined = 0
        return

    def get_renderer(self):
        return mujoco_py.MjViewer(self.sim)

    def reset(self):
        self.episode_state = 0
        self.view_front = []
        self.view_back = []
        self.time = 0
        self.joined = 0
        self.sim.reset()
        self.qpos_1 = self.sim.data.get_joint_qpos("bot_1")
        self.qpos_2 = self.sim.data.get_joint_qpos("bot_2")
        self.qvel_1 = self.sim.data.get_joint_qvel("bot_1")
        self.qvel_2 = self.sim.data.get_joint_qvel("bot_2")
        return

    def randomize(self):
        pos1 = list(np.random.uniform(low=-5, high=5, size=(2, 1)))
        pos2 = list(np.random.uniform(low=-5, high=5, size=(2, 1)))
        ori1 = list(Quaternion(axis=[0, 0, 1], angle = np.random.uniform(low=0, high=3.1415)))
        ori2 = list(Quaternion(axis=[0, 0, 1], angle = np.random.uniform(low=0, high=3.1415)))
        # pos1 = pos[0:2]
        self.sim.data.set_joint_qpos("bot_1", pos1+[1]+ori1)
        self.sim.data.set_joint_qpos("bot_2", pos2+[2]+ori2)
        self.qpos_1 = self.sim.data.get_joint_qpos("bot_1")
        self.qpos_2 = self.sim.data.get_joint_qpos("bot_2")
        self.qvel_1 = self.sim.data.get_joint_qvel("bot_1")
        self.qvel_2 = self.sim.data.get_joint_qvel("bot_2")
        return

    def evaluate(self):
        if self.episode_state == 1:
            return 10
        else:
            return -1
    def check_contact(self):
        c1 = self.sim.data.sensordata[0] >= 0
        c2 = ((self.qpos_1[1]-self.qpos_2[1])<0.255)*((self.qpos_1[1]-self.qpos_2[1])>0)
        c3 = (np.sum((self.qpos_1[-4:]-self.qpos_2[-4:]))<0.001)
        c4 = (np.sum(self.qpos_1[:3]-self.qpos_2[:3])<0.26)
        return c1*c2*c3*c4*(self.joined==0)
    def step(self, action):
        t = 0
        # contact = 0
        steps = self.ctrl_time_step/self.sim_time_step
        while t <= steps:
            self.sim.data.ctrl[:2] = action[:2]
            self.sim.data.ctrl[7:9] = action[2:4]
            if ((self.check_contact())):
                self.joined = 1
            if self.joined:
                self.join_action()
                self.episode_state = 1
            else:
                self.default_action()
            self.sim.step()
            t += 1
        # self.view_front.append(self.sim.render(256,256,camera_name="camera_front_1"))
        # self.view_back.append(self.sim.render(256,256,camera_name="camera_back_2"))
        if len(self.view_front)>5:
            # del self.view_back[0]
            del self.view_front[0]
        
        self.time += self.ctrl_time_step
        self.qpos_1 = self.sim.data.get_joint_qpos("bot_1")
        self.qpos_2 = self.sim.data.get_joint_qpos("bot_2")
        self.qvel_1 = self.sim.data.get_joint_qvel("bot_1")
        self.qvel_2 = self.sim.data.get_joint_qvel("bot_2")
        out = self.evaluate()        
        return out

    def join_action(self):
        ctrl = [-0.003,-0.003,-0.005,0.01,-0.01]
        self.sim.data.ctrl[2:7]=ctrl
        ctrl = [0.005,0.005,0.003,-0.01,0.01]
        self.sim.data.ctrl[9:14]=ctrl
        return 

    def default_action(self):
        ctrl = [0.01,0.01,-0.01,-0.001,0.001]
        self.sim.data.ctrl[2:7]=ctrl
        ctrl = [0.01,0.01,-0.01,-0.001,0.001]
        self.sim.data.ctrl[9:14]=ctrl
        
        return
    
