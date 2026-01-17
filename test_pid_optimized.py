#!/usr/bin/env python3
"""
测试优化后的PID控制参数
"""

import numpy as np
import logging
from close_combat_env import CloseCombatEnv

# 设置日志级别为INFO，减少调试输出
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_optimized_pid():
    """测试优化后的PID控制参数"""
    print("=== 测试优化后的PID控制参数 ===")

    # 创建环境
    env = CloseCombatEnv(config={
        'sim_freq': 60,
        'max_steps': 200,
        'min_altitude': 200,
        'min_velocity': 100,
        'init_distance': 5000,
        'init_altitude': 5000,
        'missile_count': 2,
    })

    # 重置环境
    obs, info = env.reset()
    print(f"环境重置完成")
    print(f"初始高度: 5000m, 目标高度: 5000m, 目标速度: 250m/s")

    # 运行测试
    max_steps = 200
    print(f"\n运行 {max_steps} 步测试...")

    # 记录数据用于分析
    altitudes = []
    speeds = []
    pitches = []
    rolls = []
    throttles = []
    elevators = []

    for step in range(max_steps):
        # 使用平飞动作
        action = env.level_flight_action(env.red_aircraft, target_altitude=5000, target_speed=250.0)

        # 执行一步
        obs, reward, terminated, truncated, info = env.step(action)

        # 记录数据
        if env.red_aircraft and env.red_aircraft.is_alive:
            red_pos = env.red_aircraft.get_position()
            red_vel = env.red_aircraft.get_velocity()
            red_rpy = env.red_aircraft.get_rpy()

            altitudes.append(red_pos[2])
            speeds.append(np.linalg.norm(red_vel))
            pitches.append(np.degrees(red_rpy[1]))
            rolls.append(np.degrees(red_rpy[0]))
            throttles.append(action[0])
            elevators.append(action[1])

        # 每50步打印一次状态
        if step % 50 == 0:
            print(f"\n步 {step}:")
            if env.red_aircraft and env.red_aircraft.is_alive:
                print(f"  高度: {red_pos[2]:.1f}m (目标: 5000m)")
                print(f"  速度: {np.linalg.norm(red_vel):.1f}m/s (目标: 250m/s)")
                print(f"  俯仰角: {np.degrees(red_rpy[1]):.2f}° (目标: 0°)")
                print(f"  滚转角: {np.degrees(red_rpy[0]):.2f}° (目标: 0°)")
                print(f"  油门: {action[0]:.3f}, 升降舵: {action[1]:.3f}")

    # 分析结果
    print("\n=== 性能分析 ===")
    if altitudes:
        altitudes = np.array(altitudes)
        speeds = np.array(speeds)
        pitches = np.array(pitches)
        rolls = np.array(rolls)

        print(f"高度控制:")
        print(f"  平均高度: {np.mean(altitudes):.1f}m")
        print(f"  高度标准差: {np.std(altitudes):.1f}m")
        print(f"  最大高度偏差: {np.max(np.abs(altitudes - 5000)):.1f}m")

        print(f"\n速度控制:")
        print(f"  平均速度: {np.mean(speeds):.1f}m/s")
        print(f"  速度标准差: {np.std(speeds):.1f}m/s")
        print(f"  最大速度偏差: {np.max(np.abs(speeds - 250)):.1f}m/s")

        print(f"\n姿态控制:")
        print(f"  平均俯仰角: {np.mean(pitches):.2f}°")
        print(f"  俯仰角标准差: {np.std(pitches):.2f}°")
        print(f"  平均滚转角: {np.mean(rolls):.2f}°")
        print(f"  滚转角标准差: {np.std(rolls):.2f}°")

        print(f"\n控制输出:")
        print(f"  平均油门: {np.mean(throttles):.3f}")
        print(f"  油门标准差: {np.std(throttles):.3f}")
        print(f"  平均升降舵: {np.mean(elevators):.3f}")
        print(f"  升降舵标准差: {np.std(elevators):.3f}")

        # 评估控制性能
        altitude_rmse = np.sqrt(np.mean((altitudes - 5000) ** 2))
        speed_rmse = np.sqrt(np.mean((speeds - 250) ** 2))
        pitch_rmse = np.sqrt(np.mean(pitches ** 2))
        roll_rmse = np.sqrt(np.mean(rolls ** 2))

        print(f"\n控制性能指标 (RMSE):")
        print(f"  高度RMSE: {altitude_rmse:.1f}m")
        print(f"  速度RMSE: {speed_rmse:.1f}m/s")
        print(f"  俯仰角RMSE: {pitch_rmse:.2f}°")
        print(f"  滚转角RMSE: {roll_rmse:.2f}°")

        # 总体评分
        overall_score = 100 - (altitude_rmse * 0.1 + speed_rmse * 0.2 + pitch_rmse * 0.5 + roll_rmse * 0.5)
        overall_score = max(0, min(100, overall_score))
        print(f"\n总体控制评分: {overall_score:.1f}/100")

    print("\n测试完成")

    # 关闭环境
    env.close()

def test_disturbance_rejection():
    """测试PID控制的抗干扰能力"""
    print("\n=== 测试PID控制的抗干扰能力 ===")

    # 创建环境
    env = CloseCombatEnv(config={
        'sim_freq': 60,
        'max_steps': 300,
        'min_altitude': 200,
        'min_velocity': 100,
        'init_distance': 5000,
        'init_altitude': 5000,
        'missile_count': 2,
    })

    # 重置环境
    obs, info = env.reset()

    print("测试计划:")
    print("1. 0-100步: 正常平飞")
    print("2. 100-150步: 模拟高度扰动（目标高度变为5200m）")
    print("3. 150-200步: 模拟速度扰动（目标速度变为300m/s）")
    print("4. 200-250步: 恢复正常平飞")
    print("5. 250-300步: 结束测试")

    altitudes = []
    speeds = []

    for step in range(300):
        # 应用不同的扰动
        if step < 100:
            # 正常平飞
            target_altitude = 5000
            target_speed = 250
        elif step < 150:
            # 高度扰动
            target_altitude = 5200
            target_speed = 250
        elif step < 200:
            # 速度扰动
            target_altitude = 5000
            target_speed = 300
        elif step < 250:
            # 恢复正常
            target_altitude = 5000
            target_speed = 250
        else:
            # 结束阶段
            target_altitude = 5000
            target_speed = 250

        # 使用平飞动作
        action = env.level_flight_action(env.red_aircraft, target_altitude=target_altitude, target_speed=target_speed)

        # 执行一步
        obs, reward, terminated, truncated, info = env.step(action)

        # 记录数据
        if env.red_aircraft and env.red_aircraft.is_alive:
            red_pos = env.red_aircraft.get_position()
            red_vel = env.red_aircraft.get_velocity()

            altitudes.append(red_pos[2])
            speeds.append(np.linalg.norm(red_vel))

        # 在扰动开始和结束时打印状态
        if step in [99, 149, 199, 249, 299]:
            print(f"\n步 {step}:")
            if env.red_aircraft and env.red_aircraft.is_alive:
                print(f"  当前高度: {red_pos[2]:.1f}m (目标: {target_altitude}m)")
                print(f"  当前速度: {np.linalg.norm(red_vel):.1f}m/s (目标: {target_speed}m/s)")

    # 分析抗干扰性能
    print("\n=== 抗干扰性能分析 ===")

    # 计算各阶段的稳定时间
    def calculate_settling_time(data, target, threshold=0.01, start_idx=0):
        """计算稳定时间（达到目标值±threshold范围内）"""
        for i in range(start_idx, len(data)):
            if abs(data[i] - target) <= abs(target * threshold):
                # 检查是否持续稳定
                stable_count = 0
                for j in range(i, min(i + 10, len(data))):
                    if abs(data[j] - target) <= abs(target * threshold):
                        stable_count += 1
                if stable_count >= 5:  # 连续5步稳定
                    return i - start_idx
        return len(data) - start_idx

    # 高度扰动响应
    altitude_settling = calculate_settling_time(altitudes[100:150], 5200, 0.01, 0)
    print(f"高度扰动响应:")
    print(f"  目标高度变化: 5000m → 5200m (+200m)")
    print(f"  稳定时间: {altitude_settling}步 ({altitude_settling/60:.1f}秒)")

    # 速度扰动响应
    speed_settling = calculate_settling_time(speeds[150:200], 300, 0.01, 0)
    print(f"\n速度扰动响应:")
    print(f"  目标速度变化: 250m/s → 300m/s (+50m/s)")
    print(f"  稳定时间: {speed_settling}步 ({speed_settling/60:.1f}秒)")

    print("\n抗干扰测试完成")
    env.close()

if __name__ == "__main__":
    print("开始测试优化后的PID控制参数\n")

    # 测试1: 优化后的PID参数性能
    test_optimized_pid()

    # 测试2: 抗干扰能力
    test_disturbance_rejection()

    print("\n所有测试完成")