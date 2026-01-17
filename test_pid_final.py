#!/usr/bin/env python3
"""
最终PID控制测试
"""

import numpy as np
import logging
from close_combat_env import CloseCombatEnv

# 设置日志级别为WARNING，减少输出
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_final_pid_performance():
    """测试最终PID控制性能"""
    print("=== 最终PID控制性能测试 ===")

    # 创建环境
    env = CloseCombatEnv(config={
        'sim_freq': 60,
        'max_steps': 600,  # 10秒仿真
        'min_altitude': 200,
        'min_velocity': 100,
        'init_distance': 5000,
        'init_altitude': 5000,
        'missile_count': 2,
    })

    # 重置环境
    obs, info = env.reset()
    print("环境初始化完成")
    print("测试目标: 维持5000m高度和250m/s速度平飞")

    # 运行测试
    max_steps = 600
    print(f"运行 {max_steps} 步测试（约10秒）...")

    # 记录关键数据
    data = {
        'step': [],
        'altitude': [],
        'speed': [],
        'pitch': [],
        'roll': [],
        'throttle': [],
        'elevator': [],
        'aileron': [],
        'rudder': []
    }

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

            data['step'].append(step)
            data['altitude'].append(red_pos[2])
            data['speed'].append(np.linalg.norm(red_vel))
            data['pitch'].append(np.degrees(red_rpy[1]))
            data['roll'].append(np.degrees(red_rpy[0]))
            data['throttle'].append(action[0])
            data['elevator'].append(action[1])
            data['aileron'].append(action[3])
            data['rudder'].append(action[2])

        # 每100步打印一次简要状态
        if step % 100 == 0 and step > 0:
            if env.red_aircraft and env.red_aircraft.is_alive:
                print(f"步 {step}: 高度={red_pos[2]:.1f}m, 速度={np.linalg.norm(red_vel):.1f}m/s, "
                      f"俯仰角={np.degrees(red_rpy[1]):.2f}°")

        # 检查是否提前终止
        if terminated or truncated:
            print(f"测试提前结束于步 {step}")
            break

    # 性能分析
    print("\n=== 最终性能分析 ===")

    if data['altitude']:
        # 转换为numpy数组
        for key in data:
            if key != 'step':
                data[key] = np.array(data[key])

        # 计算统计指标
        altitude_mean = np.mean(data['altitude'])
        altitude_std = np.std(data['altitude'])
        altitude_error = np.mean(np.abs(data['altitude'] - 5000))

        speed_mean = np.mean(data['speed'])
        speed_std = np.std(data['speed'])
        speed_error = np.mean(np.abs(data['speed'] - 250))

        pitch_mean = np.mean(data['pitch'])
        pitch_std = np.std(data['pitch'])
        pitch_abs_mean = np.mean(np.abs(data['pitch']))

        roll_mean = np.mean(data['roll'])
        roll_std = np.std(data['roll'])
        roll_abs_mean = np.mean(np.abs(data['roll']))

        # 输出结果
        print(f"高度控制:")
        print(f"  平均高度: {altitude_mean:.1f}m (目标: 5000m)")
        print(f"  高度稳定性: ±{altitude_std:.1f}m (标准差)")
        print(f"  平均高度误差: {altitude_error:.1f}m")

        print(f"\n速度控制:")
        print(f"  平均速度: {speed_mean:.1f}m/s (目标: 250m/s)")
        print(f"  速度稳定性: ±{speed_std:.1f}m/s (标准差)")
        print(f"  平均速度误差: {speed_error:.1f}m/s")

        print(f"\n姿态控制:")
        print(f"  平均俯仰角: {pitch_mean:.2f}° (目标: 0°)")
        print(f"  俯仰角稳定性: ±{pitch_std:.2f}° (标准差)")
        print(f"  平均俯仰角绝对值: {pitch_abs_mean:.2f}°")

        print(f"  平均滚转角: {roll_mean:.2f}° (目标: 0°)")
        print(f"  滚转角稳定性: ±{roll_std:.2f}° (标准差)")
        print(f"  平均滚转角绝对值: {roll_abs_mean:.2f}°")

        print(f"\n控制输出统计:")
        print(f"  油门范围: {np.min(data['throttle']):.3f} - {np.max(data['throttle']):.3f}")
        print(f"  升降舵范围: {np.min(data['elevator']):.3f} - {np.max(data['elevator']):.3f}")
        print(f"  副翼范围: {np.min(data['aileron']):.3f} - {np.max(data['aileron']):.3f}")
        print(f"  方向舵范围: {np.min(data['rudder']):.3f} - {np.max(data['rudder']):.3f}")

        # 性能评估
        print(f"\n=== 性能评估 ===")

        # 高度保持性能（权重最高）
        altitude_score = max(0, 100 - altitude_error * 2)

        # 速度保持性能
        speed_score = max(0, 100 - speed_error * 4)

        # 姿态保持性能
        pitch_score = max(0, 100 - pitch_abs_mean * 10)
        roll_score = max(0, 100 - roll_abs_mean * 20)

        # 总体评分
        overall_score = (altitude_score * 0.4 + speed_score * 0.3 +
                        pitch_score * 0.2 + roll_score * 0.1)

        print(f"各项评分:")
        print(f"  高度保持: {altitude_score:.1f}/100")
        print(f"  速度保持: {speed_score:.1f}/100")
        print(f"  俯仰保持: {pitch_score:.1f}/100")
        print(f"  滚转保持: {roll_score:.1f}/100")
        print(f"\n总体性能评分: {overall_score:.1f}/100")

        # 性能等级
        if overall_score >= 90:
            performance_level = "优秀"
        elif overall_score >= 80:
            performance_level = "良好"
        elif overall_score >= 70:
            performance_level = "中等"
        elif overall_score >= 60:
            performance_level = "及格"
        else:
            performance_level = "需要改进"

        print(f"性能等级: {performance_level}")

        # 改进建议
        print(f"\n=== 改进建议 ===")
        if altitude_error > 20:
            print(f"  • 高度控制需要加强，考虑增加积分项或提高比例增益")
        if speed_error > 5:
            print(f"  • 速度控制需要加强，油门响应可能不足")
        if pitch_abs_mean > 2:
            print(f"  • 俯仰角控制需要改进，飞机俯仰波动较大")
        if roll_abs_mean > 1:
            print(f"  • 滚转角控制需要改进，机翼不够水平")

        if overall_score >= 80:
            print(f"  • PID控制性能良好，可以用于强化学习训练")
        else:
            print(f"  • 建议进一步调整PID参数或控制策略")

    print("\n测试完成")

    # 关闭环境
    env.close()

def quick_verification():
    """快速验证PID控制基本功能"""
    print("\n=== 快速验证 ===")

    # 创建简化环境
    env = CloseCombatEnv(config={
        'sim_freq': 60,
        'max_steps': 60,  # 1秒测试
        'min_altitude': 200,
        'min_velocity': 100,
        'init_distance': 5000,
        'init_altitude': 5000,
        'missile_count': 2,
    })

    # 重置环境
    obs, info = env.reset()

    print("运行1秒（60步）快速测试...")

    initial_altitude = None
    initial_speed = None
    final_altitude = None
    final_speed = None

    for step in range(60):
        action = env.level_flight_action(env.red_aircraft, target_altitude=5000, target_speed=250.0)
        obs, reward, terminated, truncated, info = env.step(action)

        if env.red_aircraft and env.red_aircraft.is_alive:
            red_pos = env.red_aircraft.get_position()
            red_vel = env.red_aircraft.get_velocity()

            if step == 0:
                initial_altitude = red_pos[2]
                initial_speed = np.linalg.norm(red_vel)
            if step == 59:
                final_altitude = red_pos[2]
                final_speed = np.linalg.norm(red_vel)

    if initial_altitude is not None and final_altitude is not None:
        print(f"初始状态: 高度={initial_altitude:.1f}m, 速度={initial_speed:.1f}m/s")
        print(f"最终状态: 高度={final_altitude:.1f}m, 速度={final_speed:.1f}m/s")

        altitude_change = final_altitude - initial_altitude
        speed_change = final_speed - initial_speed

        print(f"高度变化: {altitude_change:+.1f}m")
        print(f"速度变化: {speed_change:+.1f}m/s")

        if abs(altitude_change) < 10 and abs(speed_change) < 5:
            print("✓ PID控制基本功能正常")
        else:
            print("⚠ PID控制需要调整")

    env.close()

if __name__ == "__main__":
    print("开始最终PID控制测试\n")

    # 测试1: 完整性能测试
    test_final_pid_performance()

    # 测试2: 快速验证
    quick_verification()

    print("\n所有测试完成")