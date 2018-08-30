import mujoco_py
from dcmr import *
from PIL import Image
import numpy as np


def display(data):
    img = Image.fromarray(data, 'RGB')
    # img.save('my.png')
    img.show()


envn = env("./xmls/two_modules.xml", 0.0001, 0.0001, 60)
r = envn.get_renderer()
t = 0
while t >= 0:
    if t<40000:
        envn.step([1, 1, 0, 0])
    else :
        envn.step([1, 0, 1 ,0])
    print (np.asarray(envn.view_back[0].shape))
    # print ((envn.qpos_1)-(envn.qpos_2))
    # print (envn.check_contact())
    # if t>15000:
    #     envn.step([1, 1, 1, 1])
    # else:
    #     envn.step([1, 1, 0, 0])


 # print (envn.qpos_1)
    # if t == 100:
    #     display(envn.view_front[4])
    # if t*envn.ctrl_time_step > 60:
    #     break
    # print (t*envn.ctrl_time_step)
    # if t%10000 == 0:
    #     envn.randomize()

    # r.render()
    t += 1
