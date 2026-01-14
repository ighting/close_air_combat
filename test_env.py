"""
测试近距格斗环境
"""
import sys
import traceback
import numpy as np
from close_combat_env import CloseCombatEnv


def test_environment():
    print("=== 测试近距格斗环境 ===")

    try:
        # 创建环境
        env = CloseCombatEnv()
        print("[OK] 环境创建成功")

        # 重置环境
        obs, info = env.reset()
        env.render()

        # 测试多个步骤
        total_reward = 0
        for i in range(10000):
            # 采样动作
            action = env.action_space.sample()

            # 执行步骤
            obs, reward, done, truncated, info = env.step(action)
            env.render()

            total_reward += reward

            if done or truncated:
                print(f"  回合结束: {info}")
                obs, info = env.reset()
                print(f"  新回合开始")

        print(f"\n总奖励: {total_reward:.4f}")

        # 关闭环境
        env.close()
        print("[OK] 环境关闭成功")

        print("\n=== 所有测试通过 ===")
        return True

    except Exception as e:
        print(f"[ERROR] 测试失败: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success1 = test_environment()
