from absl import app
from absl import flags

FLAGS = flags.FLAGS
# 仿真开始随机跳过的时间范围
flags.DEFINE_integer('skip_range', 50, 'time(seconds) range for skip randomly at the beginning')
# 每次episode的总仿真时间
flags.DEFINE_float('simulation_time', 10000, 'time for simulation')
# 黄灯的持续时间
flags.DEFINE_integer('yellow_time', 2, 'time for yellow phase')
# 计算奖励的时间间隔，动作的奖励无法立即体现
flags.DEFINE_integer('delta_rs_update_time', 10, 'time for calculate reward')
# 路网文件
flags.DEFINE_string('net_file', 'nets/osm.net.xml.gz', '')
# 车辆路由文件
flags.DEFINE_string('route_file', 'nets/osm.passenger.rou.xml', '')
# 是否使用GUI
flags.DEFINE_bool('use_gui', False, 'use sumo-gui instead of sumo')
# 训练中的episodes总数
flags.DEFINE_integer('num_episodes', 301, '')
# 使用的网络类型
flags.DEFINE_string('network', 'dqn', '')

flags.DEFINE_string('mode', 'train', '')
# ε-贪心策略的初始探索率
flags.DEFINE_float('eps_start', 1.0, '')
# 最小探索率
flags.DEFINE_float('eps_end', 0.1, '')
# ε 衰减的步数（指数衰减公式中的分母）
flags.DEFINE_integer('eps_decay', 83000, '')
# 目标网络的更新频率
flags.DEFINE_integer('target_update', 3000, '')

flags.DEFINE_string('network_file', '', '')
# 折扣因子 γ
flags.DEFINE_float('gamma', 0.95, '')
# 经验回放中采样的批次大小
flags.DEFINE_integer('batch_size', 32, '')
