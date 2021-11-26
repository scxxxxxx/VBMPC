from robot_policy.mpc.mpc_common.mppi_policy import IMPPIPolicy
from robot_policy.mpc.mpc_common.ilqr import iLQR
from robot_policy.mpc.mpc_common.shooting import Shooting


def mpc_factory(method_name: str):
    methods = {
        'ilqr': iLQR,
        'shooting': Shooting,
        'IMPPIPolicy': IMPPIPolicy
    }

    return methods[method_name]
