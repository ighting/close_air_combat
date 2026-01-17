#!/usr/bin/env python3
"""
简化PID控制测试
"""

import numpy as np
import logging
from close_combat_env import CloseCombatEnv

# 设置日志级别为ERROR，减少输出
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_simple():
    """简化测试"""
    print("=== PID控制简化测试 ===")

    # 创建环境
    env = CloseCombatEnv(config={
        'sim_freq': 60,
        'max_steps': 300,  # 5秒
        'min_altitude': 200,
        'min_velocity': 100,
        'init_distance': 5000,
        'init_altitude': 5000,
        'missile_count': 2,
    })

    # 重置环境
    obs, info = env.reset()
    print("测试开始: 维持5000m高度和250m/s速度")

    # 记录初始状态
    initial_data = []
    for step in range(10):  # 记录前10步
        action = env.level_flight_action(env.red_aircraft, target_altitude=5000, target_speed=250.0)
        obs, reward, terminated, truncated, info = env.step(action)

        if env.red_aircraft and env.red_aircraft.is_alive:
            red_pos = env.red_aircraft.get_position()
            red_vel = env.red_aircraft.get_velocity()
            red_rpy = env.red_aircraft.get_rpy()

            initial_data.append({
                'step': step,
                'altitude': red_pos[2],
                'speed': np.linalg.norm(red_vel),
                'pitch': np.degrees(red_rpy[1]),
                'throttle': action[0],
                'elevator': action[1]
            })

    print("\n前10步状态:")
    for data in initial_data:
        print(f"步 {data['step']}: 高度={data['altitude']:.1f}m, "
              f"速度={data['speed']:.1f}m/s, "
              f"俯仰角={data['pitch']:.2f}°, "
              f"油门={data['throttle']:.3f}, "
              f"升降舵={data['elevator']:.3f}")

    # 继续运行到100步
    for step in range(10, 100):
        action = env.level_flight_action(env.red_aircraft, target_altitude=5000, target_speed=250.0)
        obs, reward, terminated, truncated, info = env.step(action)

    # 记录100步状态
    if env.red_aircraft and env.red_aircraft.is_alive:
        red_pos = env.red_aircraft.get_position()
        red_vel = env.red_aircraft.get_velocity()
        red_rpy = env.red_aircraft.get_rpy()

        print(f"\n100步状态:")
        print(f"高度: {red_pos[2]:.1f}m (变化: {red_pos[2] - initial_data[0]['altitude']:+.1f}m)")
        print(f"速度: {np.linalg.norm(red_vel):.1f}m/s (变化: {np.linalg.norm(red_vel) - initial_data[0]['speed']:+.1f}m/s)")
        print(f"俯仰角: {np.degrees(red_rpy[1]):.2f}°")

    # 分析控制效果
    print("\n=== 控制效果分析 ===")

    # 计算平均油门和升降舵
    avg_throttle = np.mean([d['throttle'] for d in initial_data])
    avg_elevator = np.mean([d['elevator'] for d in initial_data])

    print(f"平均油门: {avg_throttle:.3f} (范围: 0.4-0.9)")
    print(f"平均升降舵: {avg_elevator:.3f} (范围: -0.3-0.3)")

    # 评估
    if env.red_aircraft and env.red_aircraft.is_alive:
        altitude_change = red_pos[2] - initial_data[0]['altitude']
        speed_change = np.linalg.norm(red_vel) - initial_data[0]['speed']

        if abs(altitude_change) < 50 and abs(speed_change) < 10:
            print("✓ PID控制效果良好")
        elif altitude_change < -100:
            print("⚠ 高度下降过快，需要增加油门或调整俯仰")
        elif speed_change < -20:
            print("⚠ 速度下降过快，需要增加油门")
        else:
            print("⚠ 控制效果一般，需要进一步调整")

    env.close()

if __name__ == "__main__":
    test_simple()