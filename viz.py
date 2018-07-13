from mujoco_py import load_model_from_path, MjSim, MjViewer
import os
import math

model = load_model_from_path("./DCMR.xml")
sim = MjSim(model)

viewer = MjViewer(sim)
t = 0
while True:
    sim.step()
    t = t+1
    viewer.render()
    # # if t==1000:
    #     # sim.model.eq_active[:]=True
    if t==6000:
        sim.data.ctrl[:] = 0.01
    if t==10000:
        sim.data.ctrl[:] = -0.01
        


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