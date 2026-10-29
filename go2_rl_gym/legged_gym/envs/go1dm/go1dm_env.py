"""go1dm（真狗尺寸 Go1）环境类。

关节结构（HAA/HFE/KFE × LF/LH/RF/RH，HAA 索引 [0,3,6,9]）与 go2 完全同构，
观测构造方式也一致，所以直接继承 `Go2Robot`，不需要重写任何东西。
保留这个子类是为了：
  ① 以后要做 go1dm 专属的奖励/观测时有个落点；
  ② 日志和调试时类名能区分开。
"""

from legged_gym.envs.go2.go2_env import Go2Robot


class Go1DMRobot(Go2Robot):
    pass
