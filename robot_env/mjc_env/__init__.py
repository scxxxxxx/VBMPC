from gym.envs.registration import register
from robot_env.mjc_env.armar4 import *
from robot_env.mjc_env.armar6 import *

register(id='Armar4-v0',
         entry_point='robot_env.mjc_env.armar4.armar4_env:Armar4Env',
         max_episode_steps=5000,
         reward_threshold=5000
         )

register(id='Armar4Standing-v0',
         entry_point='robot_env.mjc_env.armar4.armar4_standing_env:Armar4StandingEnv',
         max_episode_steps=5000,
         reward_threshold=5000
         )

register(id='Cassie-v0',
         entry_point='robot_env.mjc_env.cassie.cassie_env:CassieEnv',
         max_episode_steps=5000,
         reward_threshold=5000
         )

register(id='Armar6RightArm-v0',
         entry_point='robot_env.mjc_env.armar6.armar6_right_arm_env:Armar6RightArmEnv',
         max_episode_steps=5000,
         reward_threshold=5000
         )

register(id='Armar6LeftArm-v0',
         entry_point='robot_env.mjc_env.armar6.armar6_left_arm_env:Armar6LeftArmEnv',
         max_episode_steps=5000,
         reward_threshold=5000
         )

register(id='Armar6Bimanual-v0',
         entry_point='robot_env.mjc_env.armar6.armar6_bimanual_env:Armar6BimanualEnv',
         max_episode_steps=50000000,
         reward_threshold=5000
         )

register(id='Armar6BimanualTable-v0',
         entry_point='robot_env.mjc_env.armar6.armar6_bimanual_table_env:Armar6BimanualTableEnv',
         max_episode_steps=50000000,
         reward_threshold=5000
         )

register(id='Panda-v0',
         entry_point='robot_env.mjc_env.panda.panda_env:PandaEnv',
         max_episode_steps=50000000,
         reward_threshold=5000
         )

register(id='DoublePendulum-v0',
         entry_point='robot_env.mjc_env.basics.double_pendulum_env:DoublePendulum',
         max_episode_steps=50000000,
         reward_threshold=5000
         )

register(id='TriplePendulum-v0',
         entry_point='robot_env.mjc_env.basics.triple_pendulum_env:TriplePendulum',
         max_episode_steps=50000000,
         reward_threshold=5000
         )

register(id='QuadruplePendulum-v0',
         entry_point='robot_env.mjc_env.basics.quadruple_pendulum_env:QuadruplePendulum',
         max_episode_steps=50000000,
         reward_threshold=5000
         )

register(id='Pendulum-v1',
         entry_point='robot_env.mjc_env.basics.pendulum_env:Pendulum',
         max_episode_steps=50000000,
         reward_threshold=5000
         )

# to register other environments:
# register(id='Armar4-v0',
#          entry_point='robot_env.mjc_env.armar4.armar4_env:Armar4Env',
#          max_episode_steps=5000,
#          reward_threshold=5000
#          )
