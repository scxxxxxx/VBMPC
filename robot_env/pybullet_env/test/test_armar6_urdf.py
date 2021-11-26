import pybullet as p
import pybullet_data

from robot_env.utils.utils import get_robot_path, convert_mjcf_to_urdf


file = get_robot_path("armar6-urdf/armar6.urdf", "pybullet")
mjcf = "projects/control/robot-environments/robot_model/mujoco/armar6-mujoco/robot/Armar6-SH-Full.xml"
out_path = "projects/control/robot-environments/robot_model/mujoco/armar6-mujoco/robot/urdf"

# convert_mjcf_to_urdf(mjcf, os.path.join(get_home(), out_path))

physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())   # used by loadURDF
p.setGravity(0, 0, -10)
planeId = p.loadURDF("plane.urdf")
cubeStartPos = [0, 0, 1]
cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
boxId = p.loadURDF(file, cubeStartPos, cubeStartOrientation)


for i in range(100000000):
    p.stepSimulation()
    cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
    print(cubePos, cubeOrn)

p.disconnect()
