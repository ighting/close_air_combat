#!/usr/bin/env python3
"""
测试PID控制的平飞功能
"""

import numpy as np
import logging
from close_combat_env import CloseCombatEnv, PIDController

# 设置日志级别为DEBUG，以便查看PID控制信息
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_pid_controller():
    """测试PID控制器基本功能"""
    print("=== 测试PID控制器基本功能 ===")

    pid = PIDController(kp=1.0, ki=0.1, kd=0.01)

    # 模拟一个控制过程
    target = 100.0
    current = 0.0
    dt = 0.1

    print(f"目标值: {target}")
    for i in range(20):
        error = target - current
        control = pid.update(error, dt)
        current += control * 10.0  # 简单模拟系统响应

        print(f"步 {i+1}: 误差={error:.2f}, 控制输出={control:.3f}, 当前值={current:.2f}")

    print("PID控制器测试完成\n")

def test_level_flight():
    """测试平飞控制"""
    print("=== 测试平飞控制 ===")

    # 创建环境
    env = CloseCombatEnv(config={
        'sim_freq': 60,
        'max_steps': 100,
        'min_altitude': 200,
        'min_velocity': 100,
        'init_distance': 5000,
        'init_altitude': 5000,
        'missile_count': 2,
    })

    # 重置环境
    obs, info = env.reset()
    print(f"环境重置完成，初始步数: {info['step']}")
    print(f"红色飞机剩余导弹: {info['red_missiles_left']}")
    print(f"蓝色飞机剩余导弹: {info['blue_missiles_left']}")

    # 运行几步测试
    max_steps = 50
    print(f"\n运行 {max_steps} 步测试...")

    for step in range(max_steps):
        # 使用平飞动作
        action = env.level_flight_action(env.red_aircraft, target_altitude=5000, target_speed=250.0)

        # 执行一步
        obs, reward, terminated, truncated, info = env.step(action)

        # 每10步打印一次状态
        if step % 10 == 0:
            print(f"\n步 {step}:")
            print(f"  奖励: {reward:.3f}")
            print(f"  终止: {terminated}, 截断: {truncated}")
            print(f"  获胜者: {info['winner']}")
            print(f"  红色存活: {info['red_alive']}, 蓝色存活: {info['blue_alive']}")

            # 获取飞机状态
            if env.red_aircraft and env.red_aircraft.is_alive:
                red_pos = env.red_aircraft.get_position()
                red_vel = env.red_aircraft.get_velocity()
                red_rpy = env.red_aircraft.get_rpy()
                print(f"  红色飞机:")
                print(f"    高度: {red_pos[2]:.1f}m")
                print(f"    速度: {np.linalg.norm(red_vel):.1f}m/s")
                print(f"    俯仰角: {np.degrees(red_rpy[1]):.1f}°")
                print(f"    滚转角: {np.degrees(red_rpy[0]):.1f}°")

    print("\n平飞测试完成")

    # 关闭环境
    env.close()

def analyze_pid_parameters():
    """分析PID参数对控制效果的影响"""
    print("=== 分析PID参数 ===")

    # 测试不同的PID参数组合
    pid_configs = [
        {"name": "仅P控制", "kp": 0.001, "ki": 0.0, "kd": 0.0},
        {"name": "PI控制", "kp": 0.001, "ki": 0.0001, "kd": 0.0},
        {"name": "PID控制", "kp": 0.001, "ki": 0.0001, "kd": 0.0005},
        {"name": "强P控制", "kp": 0.005, "ki": 0.0, "kd": 0.0},
        {"name": "强I控制", "kp": 0.0005, "ki": 0.001, "kd": 0.0},
    ]

    for config in pid_configs:
        print(f"\n配置: {config['name']}")
        print(f"  kp={config['kp']}, ki={config['ki']}, kd={config['kd']}")

        pid = PIDController(kp=config['kp'], ki=config['ki'], kd=config['kd'])

        # 模拟高度控制
        target_altitude = 5000
        current_altitude = 4800  # 初始高度偏低
        dt = 1/60

        errors = []
        controls = []

        for i in range(100):
            error = target_altitude - current_altitude
            control = pid.update(error, dt)

            # 简单模拟飞机响应：控制输出影响高度变化率
            altitude_rate = control * 10.0  # 控制输出转换为高度变化率
            current_altitude += altitude_rate * dt

            errors.append(abs(error))
            controls.append(control)

            if i % 20 == 0:
                print(f"    步 {i}: 误差={error:.1f}m, 控制={control:.4f}, 高度={current_altitude:.1f}m")

        avg_error = np.mean(errors)
        max_control = np.max(np.abs(controls))
        print(f"  平均绝对误差: {avg_error:.1f}m")
        print(f"  最大控制输出: {max_control:.4f}")

if __name__ == "__main__":
    print("开始测试PID控制的平飞功能\n")

    # 测试1: PID控制器基本功能
    test_pid_controller()

    # 测试2: 分析PID参数
    analyze_pid_parameters()

    # 测试3: 平飞控制（需要JSBSim环境）
    try:
        test_level_flight()
    except Exception as e:
        print(f"\n注意: 平飞控制测试需要JSBSim环境，可能无法完全运行")
        print(f"错误信息: {e}")

    print("\n所有测试完成")