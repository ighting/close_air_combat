"""
红蓝双方近距格斗强化学习训练环境
基于simulator.py中的导弹和飞机类
环境要求：红蓝双方近距格斗，使用导弹攻击对方，被击中或者失速或高度太低，认为本回合结束
"""

from catalog import Catalog
from simulatior import AircraftSimulator, MissileSimulator
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import logging
from typing import Dict, Tuple, List, Optional
import copy
from dataclasses import dataclass, field


@dataclass
class PIDController:
    """PID控制器"""
    kp: float = 1.0  # 比例增益
    ki: float = 0.0  # 积分增益
    kd: float = 0.0  # 微分增益
    integral: float = 0.0  # 积分项
    prev_error: float = 0.0  # 上一次误差
    output_min: float = -1.0  # 输出最小值
    output_max: float = 1.0  # 输出最大值
    integral_min: float = -10.0  # 积分项最小值
    integral_max: float = 10.0  # 积分项最大值

    def reset(self):
        """重置控制器状态"""
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, error: float, dt: float = 1.0) -> float:
        """
        更新PID控制器

        Args:
            error: 当前误差
            dt: 时间步长（秒）

        Returns:
            control_output: 控制输出
        """
        # 比例项
        p_term = self.kp * error

        # 积分项（带抗饱和）
        self.integral += error * dt
        self.integral = np.clip(self.integral, self.integral_min, self.integral_max)
        i_term = self.ki * self.integral

        # 微分项
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        d_term = self.kd * derivative

        # 计算总输出
        output = p_term + i_term + d_term

        # 更新上一次误差
        self.prev_error = error

        # 限制输出范围
        return np.clip(output, self.output_min, self.output_max)


# TacView支持
try:
    from TacviewClient import TacviewClient
    TACVIEW_AVAILABLE = True
except ImportError:
    TACVIEW_AVAILABLE = False
    logging.warning("TacView客户端模块不可用，将禁用TacView实时通信功能")

# 导入simulator.py中的类
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'envs/JSBSim/core'))


class CloseCombatEnv(gym.Env):
    """
    红蓝双方近距格斗训练环境

    观察空间：包含飞机状态、相对位置、导弹状态等信息
    动作空间：飞机控制输入（油门、升降舵、方向舵、副翼等）
    终止条件：被导弹击中、失速、高度太低、时间限制
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, config: Dict = None):
        """
        初始化环境

        Args:
            config: 配置字典，包含环境参数
        """
        super(CloseCombatEnv, self).__init__()

        # 默认配置
        self.config = {
            'sim_freq': 60,  # 仿真频率 Hz
            'max_steps': 5000,  # 最大步数
            'min_altitude': 200,  # 最低安全高度 (m)
            'min_velocity': 100,  # 最低安全速度 (m/s)
            'init_distance': 5000,  # 初始距离 (m)
            'init_altitude': 5000,  # 初始高度 (m)
            'missile_count': 2,  # 每架飞机导弹数量
            'reward_hit': 100,  # 击中奖励
            'reward_miss': -10,  # 发射导弹未命中惩罚
            'reward_survive': 0.1,  # 生存奖励
            'reward_crash': -100,  # 坠毁惩罚
            'reward_altitude': 0.01,  # 高度保持奖励系数
            'reward_velocity': 0.01,  # 速度保持奖励系数
            # TacView配置
            'tacview_enabled': True,  # 是否启用TacView
            'tacview_host': '127.0.0.1',  # TacView监听地址
            'tacview_port': 15502,  # TacView监听端口
            'tacview_frequency': 10.0,  # 数据发送频率(Hz)
            'tacview_render_mode': 'both',  # 渲染模式: 'tacview', 'console', 'both'
        }

        if config:
            self.config.update(config)

        # 初始化飞机和导弹
        self.red_aircraft = None
        self.blue_aircraft = None
        self.missiles = []  # 所有活跃导弹

        # 初始化日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # TacView管理器
        self.tacview_manager = None
        self._init_tacview()

        # 环境状态
        self.current_step = 0
        self.done = False
        self.winner = None  # 'red', 'blue', 'draw', None

        # PID控制器（最终优化参数）
        self.altitude_pid = PIDController(kp=0.005, ki=0.001, kd=0.003, output_min=-0.3, output_max=0.3)
        self.speed_pid = PIDController(kp=0.05, ki=0.01, kd=0.03, output_min=-0.2, output_max=0.2)  # 调整为相对基础油门的偏移量
        self.pitch_pid = PIDController(kp=1.2, ki=0.3, kd=0.4, output_min=-0.3, output_max=0.3)
        self.roll_pid = PIDController(kp=1.2, ki=0.3, kd=0.4, output_min=-0.3, output_max=0.3)

        # 定义观察空间
        # 观察包含：自身状态(13) + 相对状态(6) + 导弹警告(3) = 22维
        # 自身状态: 位置(3), 速度(3), 姿态(3), 角速度(3), 剩余导弹(1)
        # 相对状态: 相对位置(3), 相对速度(3)
        # 导弹警告: 最近导弹距离(1), 方位角(1), 俯仰角(1)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(22,),
            dtype=np.float32
        )

        # 定义动作空间 (连续控制)
        # 动作: [油门, 升降舵, 方向舵, 副翼, 发射导弹]
        # 油门: 0-1, 升降舵: -1到1, 方向舵: -1到1, 副翼: -1到1, 发射导弹: 0或1
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0, -1.0, -1.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )

    def _init_tacview(self):
        """初始化TacView管理器"""
        if not TACVIEW_AVAILABLE:
            self.logger.warning("TacView客户端不可用，将禁用TacView功能")
            return

        if not self.config.get('tacview_enabled', True):
            self.logger.info("TacView功能已禁用")
            return

        try:
            # 获取当前时间作为参考时间
            import time
            current_time = time.strftime("%Y %m %d %H %M %S")

            self.tacview_manager = TacviewClient(
                serverip=self.config.get('tacview_host', '127.0.0.1'),
                serverport=self.config.get('tacview_port', 15502),
                time_str=current_time
            )

            # 建立连接
            if self.tacview_manager.connect():
                self.logger.info(
                    f"TacView客户端已连接: {self.config.get('tacview_host', '127.0.0.1')}:{self.config.get('tacview_port', 15502)}")
            else:
                self.logger.warning("TacView连接失败，将在render时尝试重连")
                self.tacview_manager = None

        except Exception as e:
            self.logger.error(f"初始化TacView管理器失败: {e}")
            self.tacview_manager = None

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        重置环境到初始状态

        Args:
            seed: 随机种子
            options: 重置选项

        Returns:
            observation: 初始观察
            info: 额外信息
        """
        if seed is not None:
            self.seed(seed)

        self.current_step = 0
        self.done = False
        self.winner = None
        self.missiles = []

        # 重置PID控制器
        self._reset_pid_controllers()

        # 清理TacView对象
        if self.tacview_manager:
            self.tacview_manager.clear_all()

        # 创建红蓝飞机
        self._create_aircraft()

        # 设置初始位置
        self._set_initial_positions()

        # 获取初始观察
        obs = self._get_observation('red')

        info = {
            'step': self.current_step,
            'red_missiles_left': self.red_aircraft.num_left_missiles,
            'blue_missiles_left': self.blue_aircraft.num_left_missiles,
        }

        return obs, info

    def _reset_pid_controllers(self):
        """重置所有PID控制器"""
        self.altitude_pid.reset()
        self.speed_pid.reset()
        self.pitch_pid.reset()
        self.roll_pid.reset()
        self.logger.debug("PID控制器已重置")

    def _create_aircraft(self):
        """创建红蓝双方飞机"""
        # 红色飞机
        red_init_state = {
            'ic_long_gc_deg': 120.0,
            'ic_lat_geod_deg': 60.0,
            'ic_h_sl_ft': self.config['init_altitude'] * 3.28084,  # 转换为英尺
            'ic_psi_true_deg': 0.0,
            'ic_u_fps': 250.0 * 3.28084,  # 转换为英尺/秒
            'ic_v_fps': 0.0,
            'ic_w_fps': 0.0,
        }

        # 蓝色飞机
        blue_init_state = {
            'ic_long_gc_deg': 120.0,
            'ic_lat_geod_deg': 60.0,
            'ic_h_sl_ft': self.config['init_altitude'] * 3.28084,
            'ic_psi_true_deg': 180.0,
            'ic_u_fps': 250.0 * 3.28084,
            'ic_v_fps': 0.0,
            'ic_w_fps': 0.0,
        }

        self.red_aircraft = AircraftSimulator(
            uid="R0100",
            color="Red",
            model='f16',
            init_state=red_init_state,
            origin=(120.0, 60.0, 0.0),
            sim_freq=self.config['sim_freq'],
            num_missiles=self.config['missile_count']
        )

        self.blue_aircraft = AircraftSimulator(
            uid="B0100",
            color="Blue",
            model='f16',
            init_state=blue_init_state,
            origin=(120.0, 60.0, 0.0),
            sim_freq=self.config['sim_freq'],
            num_missiles=self.config['missile_count']
        )

        # 设置敌对关系
        self.red_aircraft.enemies = [self.blue_aircraft]
        self.blue_aircraft.enemies = [self.red_aircraft]

    def _set_initial_positions(self):
        """设置初始位置，使两机相对飞行"""
        # 红色飞机在原点
        red_pos = np.array([0.0, 0.0, self.config['init_altitude']])

        # 蓝色飞机在红色飞机前方指定距离处
        blue_pos = np.array([
            self.config['init_distance'],
            0.0,
            self.config['init_altitude']
        ])

        # 更新飞机位置（简化实现，实际需要调用JSBSim接口）
        # 这里使用简化设置，实际项目中需要调用simulator的相应方法

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        执行一步动作

        Args:
            action: 动作向量 [油门, 升降舵, 方向舵, 副翼, 发射导弹]

        Returns:
            observation: 新的观察
            reward: 奖励值
            terminated: 是否终止（任务完成）
            truncated: 是否截断（时间限制等）
            info: 额外信息
        """
        # 解析动作
        action = self.level_flight_action(
            self.red_aircraft, target_altitude=5000)
        throttle, elevator, rudder, aileron, fire_missile = action

        # 应用控制输入到红色飞机（蓝色飞机由环境控制或对手控制）
        self._apply_control(self.red_aircraft, throttle,
                            elevator, rudder, aileron)

        # 蓝色飞机使用简单策略
        blue_action = self._get_blue_action()
        # 只传递前4个控制参数（油门、升降舵、方向舵、副翼）
        self._apply_control(self.blue_aircraft, *blue_action[:4])

        # 检查是否发射导弹
        if fire_missile > 0.5 and self.red_aircraft.num_left_missiles > 0:
            self._fire_missile(self.red_aircraft, self.blue_aircraft)

        # 运行仿真一步
        self._run_simulation_step()

        # 更新导弹状态
        self._update_missiles()

        # 检查终止条件
        terminated, self.winner = self._check_termination()

        # 计算奖励
        reward = self._calculate_reward()

        # 获取观察
        obs = self._get_observation('red')

        # 更新步数
        self.current_step += 1

        # 检查步数限制（截断条件）
        truncated = False
        if self.current_step >= self.config['max_steps']:
            truncated = True
            self.logger.info(
                f"截断条件：达到最大步数限制（{self.current_step} >= {self.config['max_steps']}），平局")
            if self.winner is None:
                self.winner = 'draw'

        info = {
            'winner': self.winner,
            'step': self.current_step,
            'red_missiles_left': self.red_aircraft.num_left_missiles,
            'blue_missiles_left': self.blue_aircraft.num_left_missiles,
            'red_alive': self.red_aircraft.is_alive,
            'blue_alive': self.blue_aircraft.is_alive,
        }

        return obs, reward, terminated, truncated, info

    def _apply_control(self, aircraft: AircraftSimulator, throttle: float,
                       elevator: float, rudder: float, aileron: float):
        """应用控制输入到飞机"""
        # 使用Catalog中的属性对象
        # 设置油门
        aircraft.set_property_value(Catalog.fcs_throttle_cmd_norm, throttle)

        # 设置升降舵
        aircraft.set_property_value(Catalog.fcs_elevator_cmd_norm, elevator)

        # 设置方向舵
        aircraft.set_property_value(Catalog.fcs_rudder_cmd_norm, rudder)

        # 设置副翼
        aircraft.set_property_value(Catalog.fcs_aileron_cmd_norm, aileron)

    def _get_blue_action(self) -> np.ndarray:
        """获取蓝色飞机的动作（简单AI）"""
        # 简单策略：保持高度和速度，尝试接近红色飞机
        red_pos = self.red_aircraft.get_position()
        blue_pos = self.blue_aircraft.get_position()

        # 计算相对位置
        rel_pos = red_pos - blue_pos
        distance = np.linalg.norm(rel_pos)

        # 简单控制逻辑
        if distance > 3000:
            # 远距离：加速接近
            throttle = 0.8
        else:
            # 近距离：保持速度
            throttle = 0.6

        # 简单的姿态控制
        elevator = 0.0
        rudder = 0.0
        aileron = 0.0

        # 随机决定是否发射导弹（有一定概率）
        fire_missile = 0.0
        if (self.blue_aircraft.num_left_missiles > 0 and
            distance < 2000 and
                np.random.random() < 0.01):
            fire_missile = 1.0

        return np.array([throttle, elevator, rudder, aileron, fire_missile])

    def level_flight_action(self, aircraft, target_altitude=None, target_speed=250.0) -> np.ndarray:
        """
        生成平飞动作（使用PID控制）

        Args:
            target_altitude: 目标高度 (m)，None表示保持当前高度
            target_speed: 目标速度 (m/s)

        Returns:
            action: 动作向量 [油门, 升降舵, 方向舵, 副翼, 发射导弹]
        """
        # 获取当前飞机状态
        if aircraft is None:
            return np.array([0.6, 0.0, 0.0, 0.0, 0.0])

        current_pos = aircraft.get_position()
        current_vel = aircraft.get_velocity()
        current_rpy = aircraft.get_rpy()

        # 当前高度和速度
        current_altitude = current_pos[2]
        current_speed = np.linalg.norm(current_vel)

        # 俯仰角和滚转角
        current_pitch = current_rpy[1]
        current_roll = current_rpy[0]

        # 计算时间步长（基于仿真频率）
        dt = 1.0 / self.config['sim_freq']

        # 1. 油门控制：PID控制速度
        speed_error = target_speed - current_speed
        pid_output = self.speed_pid.update(speed_error, dt)

        # 基础油门 + PID调整
        # F-16平飞通常需要约0.6-0.7的油门
        base_throttle = 0.65
        throttle = base_throttle + pid_output

        # 确保油门在合理范围内
        throttle = np.clip(throttle, 0.4, 0.9)

        # 2. 升降舵控制：PID控制高度和俯仰角
        if target_altitude is not None:
            # 高度控制：使用PID控制高度误差
            alt_error = target_altitude - current_altitude
            altitude_control = self.altitude_pid.update(alt_error, dt)

            # 俯仰角控制：平飞时目标俯仰角为0
            pitch_error = -current_pitch
            pitch_control = self.pitch_pid.update(pitch_error, dt)

            # 组合高度控制和俯仰角控制
            elevator = altitude_control + pitch_control
        else:
            # 保持当前高度，只控制俯仰角
            pitch_error = -current_pitch
            elevator = self.pitch_pid.update(pitch_error, dt)

        # 限制升降舵输出范围
        elevator = np.clip(elevator, -0.3, 0.3)

        # 3. 方向舵控制：平飞时保持航向，使用简单的偏航阻尼
        # 获取偏航角速度（yaw rate）
        try:
            # 尝试从飞机获取角速度
            angular_vel = aircraft.get_angular_velocity()
            yaw_rate = angular_vel[2] if len(angular_vel) > 2 else 0.0
        except:
            yaw_rate = 0.0

        # 简单的偏航阻尼：抑制偏航角速度
        rudder = -yaw_rate * 0.1
        rudder = np.clip(rudder, -0.2, 0.2)

        # 4. 副翼控制：PID控制滚转角
        roll_error = -current_roll  # 目标滚转角为0（机翼水平）
        aileron = self.roll_pid.update(roll_error, dt)
        aileron = np.clip(aileron, -0.3, 0.3)

        # 5. 不发射导弹
        fire_missile = 0.0

        # 记录控制信息（调试用）
        if self.current_step % 100 == 0:
            self.logger.debug(
                f"PID控制: 高度误差={alt_error if target_altitude is not None else 'N/A':.1f}m, "
                f"速度误差={speed_error:.1f}m/s, "
                f"俯仰角={np.degrees(current_pitch):.1f}°, "
                f"滚转角={np.degrees(current_roll):.1f}°, "
                f"油门={throttle:.3f}, 升降舵={elevator:.3f}, 副翼={aileron:.3f}"
            )

        return np.array([throttle, elevator, rudder, aileron, fire_missile])

    def _fire_missile(self, shooter: AircraftSimulator, target: AircraftSimulator):
        """发射导弹"""
        if shooter.num_left_missiles <= 0:
            return

        # 创建导弹ID
        missile_id = f"{shooter.uid}M{self.config['missile_count'] - shooter.num_left_missiles + 1}"

        # 创建导弹
        missile = MissileSimulator.create(
            parent=shooter,
            target=target,
            uid=missile_id,
            missile_model="AIM-9L"
        )

        # 添加到导弹列表
        self.missiles.append(missile)

        # 减少导弹数量
        shooter.num_left_missiles -= 1

        self.logger.info(
            f"{shooter.color} aircraft fired missile {missile_id}")

    def _run_simulation_step(self):
        """运行仿真一步"""
        # 运行红色飞机
        if self.red_aircraft.is_alive:
            self.red_aircraft.run()

        # 运行蓝色飞机
        if self.blue_aircraft.is_alive:
            self.blue_aircraft.run()

        # 运行所有导弹
        for missile in self.missiles[:]:  # 使用副本遍历
            if missile.is_alive:
                missile.run()

    def _update_missiles(self):
        """更新导弹状态，移除已完成任务的导弹"""
        active_missiles = []
        for missile in self.missiles:
            if missile.is_alive:
                active_missiles.append(missile)
            elif missile.is_done:
                # 导弹已完成任务（命中或未命中）
                if missile.is_success:
                    self.logger.info(f"Missile {missile.uid} hit target!")
                else:
                    self.logger.info(f"Missile {missile.uid} missed.")
        self.missiles = active_missiles

    def _check_termination(self) -> Tuple[bool, Optional[str]]:
        """
        检查终止条件

        Returns:
            terminated: 是否终止（任务完成）
            winner: 获胜者 ('red', 'blue', 'draw', None)
        """
        # 检查是否被击中
        if not self.red_aircraft.is_alive:
            self.logger.info(f"终止条件：红色飞机被击中，蓝色获胜")
            return True, 'blue'

        if not self.blue_aircraft.is_alive:
            self.logger.info(f"终止条件：蓝色飞机被击中，红色获胜")
            return True, 'red'

        # 检查坠毁（高度太低）
        red_altitude = self.red_aircraft.get_geodetic()[2]
        blue_altitude = self.blue_aircraft.get_geodetic()[2]

        if red_altitude < self.config['min_altitude']:
            self.logger.info(
                f"终止条件：红色飞机坠毁（高度{red_altitude:.1f}m < 最低安全高度{self.config['min_altitude']}m），蓝色获胜")
            self.red_aircraft.crash()
            return True, 'blue'

        if blue_altitude < self.config['min_altitude']:
            self.logger.info(
                f"终止条件：蓝色飞机坠毁（高度{blue_altitude:.1f}m < 最低安全高度{self.config['min_altitude']}m），红色获胜")
            self.blue_aircraft.crash()
            return True, 'red'

        # 检查失速（速度太低）
        red_velocity = np.linalg.norm(self.red_aircraft.get_velocity())
        blue_velocity = np.linalg.norm(self.blue_aircraft.get_velocity())

        if red_velocity < self.config['min_velocity']:
            self.logger.info(
                f"终止条件：红色飞机失速（速度{red_velocity:.1f}m/s < 最低安全速度{self.config['min_velocity']}m/s），蓝色获胜")
            self.red_aircraft.crash()
            return True, 'blue'

        if blue_velocity < self.config['min_velocity']:
            self.logger.info(
                f"终止条件：蓝色飞机失速（速度{blue_velocity:.1f}m/s < 最低安全速度{self.config['min_velocity']}m/s），红色获胜")
            self.blue_aircraft.crash()
            return True, 'red'

        return False, None

    def _calculate_reward(self) -> float:
        """计算奖励"""
        reward = 0.0

        # 生存奖励
        if self.red_aircraft.is_alive:
            reward += self.config['reward_survive']

        # 高度保持奖励（鼓励保持安全高度）
        red_altitude = self.red_aircraft.get_geodetic()[2]
        altitude_diff = abs(red_altitude - self.config['init_altitude'])
        reward -= self.config['reward_altitude'] * altitude_diff / 1000.0

        # 速度保持奖励（鼓励保持安全速度）
        red_velocity = np.linalg.norm(self.red_aircraft.get_velocity())
        velocity_diff = abs(red_velocity - 250.0)  # 250 m/s 为目标速度
        reward -= self.config['reward_velocity'] * velocity_diff / 100.0

        # 击中奖励（如果红色导弹击中蓝色飞机）
        for missile in self.missiles:
            if (missile.parent_aircraft == self.red_aircraft and
                    missile.is_success):
                reward += self.config['reward_hit']

        # 被击中惩罚（如果蓝色导弹击中红色飞机）
        if not self.red_aircraft.is_alive and self.red_aircraft.is_shotdown:
            reward += self.config['reward_crash']

        # 发射导弹未命中惩罚
        # 这里简化处理，实际需要跟踪每发导弹的状态

        return reward

    def _get_observation(self, perspective: str = 'red') -> np.ndarray:
        """
        获取观察

        Args:
            perspective: 观察视角 ('red' 或 'blue')

        Returns:
            观察向量
        """
        if perspective == 'red':
            own = self.red_aircraft
            other = self.blue_aircraft
        else:
            own = self.blue_aircraft
            other = self.red_aircraft

        # 自身状态
        own_pos = own.get_position()
        own_vel = own.get_velocity()
        own_rpy = own.get_rpy()

        # 角速度（简化，实际应从JSBSim获取）
        own_ang_vel = np.zeros(3)

        # 相对状态
        other_pos = other.get_position()
        other_vel = other.get_velocity()

        rel_pos = other_pos - own_pos
        rel_vel = other_vel - own_vel

        # 导弹警告信息
        missile_warning = self._get_missile_warning(own)

        # 构建观察向量
        obs = np.concatenate([
            own_pos,                    # 3: 位置 (north, east, up)
            own_vel,                    # 3: 速度 (v_north, v_east, v_up)
            own_rpy,                    # 3: 姿态 (roll, pitch, yaw)
            own_ang_vel,                # 3: 角速度 (p, q, r)
            [own.num_left_missiles / self.config['missile_count']],  # 1: 剩余导弹比例
            rel_pos,                    # 3: 相对位置
            rel_vel,                    # 3: 相对速度
            missile_warning,            # 3: 导弹警告信息
        ])

        return obs.astype(np.float32)

    def _get_missile_warning(self, aircraft: AircraftSimulator) -> np.ndarray:
        """
        获取导弹警告信息

        Returns:
            [最近导弹距离, 方位角, 俯仰角]
        """
        if not aircraft.under_missiles:
            return np.array([10000.0, 0.0, 0.0])  # 无威胁

        # 找到最近的导弹
        closest_missile = None
        min_distance = float('inf')

        for missile in aircraft.under_missiles:
            if missile.is_alive:
                missile_pos = missile.get_position()
                aircraft_pos = aircraft.get_position()
                distance = np.linalg.norm(missile_pos - aircraft_pos)

                if distance < min_distance:
                    min_distance = distance
                    closest_missile = missile

        if closest_missile is None:
            return np.array([10000.0, 0.0, 0.0])

        # 计算相对方位
        missile_pos = closest_missile.get_position()
        aircraft_pos = aircraft.get_position()
        rel_pos = missile_pos - aircraft_pos

        # 计算方位角（相对于飞机机头方向）
        # 简化：使用全局坐标系
        azimuth = np.arctan2(rel_pos[1], rel_pos[0])  # 东-北平面
        elevation = np.arctan2(rel_pos[2], np.linalg.norm(rel_pos[:2]))

        return np.array([min_distance, azimuth, elevation])

    def render(self, mode='human'):
        """
        渲染环境

        支持多种渲染模式：
        - 'human': 控制台输出
        - 'tacview': TacView实时通信
        - 'both': 同时使用控制台和TacView
        """
        render_mode = self.config.get('tacview_render_mode', 'both')

        if mode == 'human':
            # 控制台渲染
            if render_mode in ['console', 'both']:
                self._render_console()

            # TacView实时通信
            if render_mode in ['tacview', 'both']:
                self._render_tacview()

    def _render_console(self):
        """控制台渲染"""
        print(f"Step: {self.current_step}")
        print(
            f"Red Aircraft: {'Alive' if self.red_aircraft.is_alive else 'Dead'}")
        print(
            f"Blue Aircraft: {'Alive' if self.blue_aircraft.is_alive else 'Dead'}")
        print(f"Red Missiles Left: {self.red_aircraft.num_left_missiles}")
        print(
            f"Blue Missiles Left: {self.blue_aircraft.num_left_missiles}")
        print(f"Active Missiles: {len(self.missiles)}")

        # 显示飞机状态详细信息
        if self.red_aircraft.is_alive:
            red_altitude = self.red_aircraft.get_geodetic()[2]
            red_velocity = np.linalg.norm(self.red_aircraft.get_velocity())
            # print(
            #     f"Red Altitude: {red_altitude:.1f}m (min: {self.config['min_altitude']}m)")
            # print(
            #     f"Red Velocity: {red_velocity:.1f}m/s (min: {self.config['min_velocity']}m/s)")

        if self.blue_aircraft.is_alive:
            blue_altitude = self.blue_aircraft.get_geodetic()[2]
            blue_velocity = np.linalg.norm(self.blue_aircraft.get_velocity())
            # print(
            #     f"Blue Altitude: {blue_altitude:.1f}m (min: {self.config['min_altitude']}m)")
            # print(
            #     f"Blue Velocity: {blue_velocity:.1f}m/s (min: {self.config['min_velocity']}m/s)")

        if self.done:
            print(f"Episode ended. Winner: {self.winner}")
            print(
                f"Steps taken: {self.current_step}/{self.config['max_steps']}")

    def _render_tacview(self):
        """TacView实时通信渲染"""
        if not self.tacview_manager:
            return

        try:
            # 更新红色飞机状态
            if self.red_aircraft and self.red_aircraft.is_alive:
                lon, lat, alt = self.red_aircraft.get_geodetic()
                roll, pitch, yaw = self.red_aircraft.get_rpy()

                # 转换为度
                roll_deg = np.degrees(roll)
                pitch_deg = np.degrees(pitch)
                yaw_deg = np.degrees(yaw)

                self.tacview_manager.update_aircraft(
                    aircraft_uid=self.red_aircraft.uid,
                    color=self.red_aircraft.color,
                    longitude=lon,
                    latitude=lat,
                    altitude=alt,
                    roll=roll_deg,
                    pitch=pitch_deg,
                    heading=yaw_deg
                )
            else:
                # 飞机被摧毁，从TacView中移除
                self.tacview_manager.remove_aircraft(self.red_aircraft.uid)

            # 更新蓝色飞机状态
            if self.blue_aircraft and self.blue_aircraft.is_alive:
                lon, lat, alt = self.blue_aircraft.get_geodetic()
                roll, pitch, yaw = self.blue_aircraft.get_rpy()

                # 转换为度
                roll_deg = np.degrees(roll)
                pitch_deg = np.degrees(pitch)
                yaw_deg = np.degrees(yaw)

                self.tacview_manager.update_aircraft(
                    aircraft_uid=self.blue_aircraft.uid,
                    color=self.blue_aircraft.color,
                    longitude=lon,
                    latitude=lat,
                    altitude=alt,
                    roll=roll_deg,
                    pitch=pitch_deg,
                    heading=yaw_deg
                )
            else:
                # 飞机被摧毁，从TacView中移除
                self.tacview_manager.remove_aircraft(self.blue_aircraft.uid)

            # 更新导弹状态
            for missile in self.missiles:
                if missile.is_alive:
                    lon, lat, alt = missile.get_geodetic()
                    roll, pitch, yaw = missile.get_rpy()

                    # 转换为度
                    roll_deg = np.degrees(roll)
                    pitch_deg = np.degrees(pitch)
                    yaw_deg = np.degrees(yaw)

                    self.tacview_manager.update_missile(
                        missile_uid=missile.uid,
                        color=missile.color,
                        longitude=lon,
                        latitude=lat,
                        altitude=alt,
                        roll=roll_deg,
                        pitch=pitch_deg,
                        heading=yaw_deg
                    )
                elif missile.is_done:
                    # 导弹已完成任务
                    if missile.is_success:
                        # 导弹命中，添加爆炸效果
                        lon, lat, alt = missile.get_geodetic()
                        self.tacview_manager.add_explosion(
                            explosion_uid=f"{missile.uid}_explosion",
                            color=missile.color,
                            longitude=lon,
                            latitude=lat,
                            altitude=alt,
                            radius=300.0  # 爆炸半径
                        )

                    # 从TacView中移除导弹
                    self.tacview_manager.remove_missile(missile.uid)

            # 清理已完成的爆炸效果（可选，可以设置过期时间）
            # 这里简化处理，爆炸效果会一直显示

        except Exception as e:
            self.logger.error(f"TacView渲染失败: {e}")

    def close(self):
        """关闭环境，清理资源"""
        # 清理TacView资源
        if self.tacview_manager:
            self.tacview_manager.clear_all()
            self.tacview_manager.close()
            self.tacview_manager = None

        # 清理飞机资源
        if self.red_aircraft:
            self.red_aircraft.close()
        if self.blue_aircraft:
            self.blue_aircraft.close()

        # 清理导弹资源
        for missile in self.missiles:
            missile.close()

        self.missiles.clear()

    def seed(self, seed=None):
        """设置随机种子"""
        np.random.seed(seed)
        return [seed]
