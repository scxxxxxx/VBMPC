import pybullet as p
import pybullet_data

# Can alternatively pass in p.DIRECT
client = p.connect(p.GUI)
p.setGravity(0, 0, -10, physicsClientId=client)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
# panda_urdf = "/home/gao/projects/control/robot-environments/robots/pybullet/franka/panda.urdf"
panda_urdf = "/home/gao/projects/control/stepping_stone/mocca_envs/data/objects/steps/plank_large.urdf"
# robotId = p.loadMJCF(panda_urdf)
pandaId = p.loadURDF("plane.urdf", basePosition=[0, 0, 0.2])
planeId = p.loadURDF(panda_urdf, basePosition=[0, 0, 2.2], useFixedBase=1)
# carId = p.loadURDF("racecar/racecar.urdf", basePosition=[0, 0, 0.2])

for _ in range(3000000):
    pos, ori = p.getBasePositionAndOrientation(pandaId)
    # p.applyExternalForce(carId, 0, [10, 0, 0], pos, p.WORLD_FRAME)
    p.stepSimulation()


