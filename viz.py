from mujoco_py import load_model_from_path, MjSim, MjViewer
import os
import math

model = load_model_from_path("./xmls/module.xml")
sim = MjSim(model)

viewer = MjViewer(sim)
t = 0
while True:
    sim.step()
    t = t+1
    # if t*0.0001 ==60:
    #     break
    sim.data.ctrl[2] = 0.01
    sim.data.ctrl[3] = 0.01
    sim.data.ctrl[4] = 0.01
    sim.data.ctrl[5] = -0.001
    sim.data.ctrl[6] = 0.001
    viewer.render()
    # if t==10000:
        # sim.data.set_joint_qpos("bot_1",[1 ,1 ,1, 1, 0 ,0, 0])
    # # if t==1000:
    # #     # sim.model.eq_active[:]=True
    # if t*0.01 == 3:
    #     # sim.data.ctrl[:] = 0.001
    #     sim.data.set_joint_qpos("body_1",[0, 0, 0, 1, 0, 0, 0])
    # if t*0.01 == 6:
    #     # sim.data.ctrl[:] = -0.001
    #     sim.data.set_joint_qpos("body_1",[1, 0, 0, 1, 0, 0, 0])

    # x = sim.data.get_joint_qpos("bot_1")
    # print(x)
    # if t*0.01 == 300:
    #     break


# sim_state = sim.get_state()

# while True:
#     sim.set_state(sim_state)

#     for i in range(1000):
#         if i < 150:
#             sim.data.ctrl[:] = 0.0
#         else:
#             sim.data.ctrl[:] = -1.0
#         sim.step()
#         viewer.render()

#     if os.getenv('TESTING') is not None:
#         break
#
