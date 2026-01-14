"""
使用Ray RLlib训练近距格斗环境
"""
# 修复NumPy 2.0兼容性问题
import numpy as np
if not hasattr(np, 'product'):
    np.product = np.prod

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.tune.registry import register_env
import logging

from close_combat_env import CloseCombatEnv, make_close_combat_env

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def env_creator(config):
    """环境创建函数，用于Ray注册"""
    return make_close_combat_env(config)

def train_ppo():
    """使用PPO算法训练"""
    # 初始化Ray
    ray.init(ignore_reinit_error=True, num_cpus=4, num_gpus=0)
    
    # 注册环境
    register_env("CloseCombatEnv", env_creator)
    
    # 配置PPO算法
    config = (
        PPOConfig()
        .environment(
            env="CloseCombatEnv",
            env_config={
                "sim_freq": 60,
                "max_steps": 1000,
                "min_altitude": 1000,
                "min_velocity": 100,
                "init_distance": 5000,
                "init_altitude": 5000,
                "missile_count": 2,
                "reward_hit": 100,
                "reward_miss": -10,
                "reward_survive": 0.1,
                "reward_crash": -100,
                "reward_altitude": 0.01,
                "reward_velocity": 0.01,
            }
        )
        .training(
            gamma=0.99,
            lr=0.0003,
            train_batch_size=4000,
            sgd_minibatch_size=256,
            num_sgd_iter=10,
            clip_param=0.2,
            model={
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "tanh",
            }
        )
        .resources(num_gpus=0)
        .rollouts(num_rollout_workers=2, rollout_fragment_length=200)
        .framework("torch")
        .debugging(log_level="INFO")
    )
    
    # 创建算法实例
    algo = config.build()
    
    # 训练循环
    max_iterations = 1000
    checkpoint_freq = 50
    
    logger.info("开始PPO训练...")
    for i in range(max_iterations):
        result = algo.train()
        
        # 打印训练指标
        episode_reward_mean = result.get("episode_reward_mean", 0)
        episode_len_mean = result.get("episode_len_mean", 0)
        
        logger.info(f"Iteration {i+1}: "
                   f"Reward mean = {episode_reward_mean:.2f}, "
                   f"Length mean = {episode_len_mean:.2f}")
        
        # 定期保存检查点
        if (i + 1) % checkpoint_freq == 0:
            checkpoint_path = algo.save(f"./checkpoints/ppo_checkpoint_{i+1}")
            logger.info(f"检查点保存到: {checkpoint_path}")
    
    # 保存最终模型
    final_checkpoint = algo.save("./checkpoints/ppo_final")
    logger.info(f"最终模型保存到: {final_checkpoint}")
    
    # 关闭Ray
    ray.shutdown()
    
    return algo

def train_sac():
    """使用SAC算法训练（适用于连续动作空间）"""
    ray.init(ignore_reinit_error=True, num_cpus=4, num_gpus=0)
    
    register_env("CloseCombatEnv", env_creator)
    
    config = (
        SACConfig()
        .environment(
            env="CloseCombatEnv",
            env_config={
                "sim_freq": 60,
                "max_steps": 1000,
            }
        )
        .training(
            gamma=0.99,
            lr=0.0003,
            train_batch_size=256,
            target_network_update_freq=1,
            tau=0.005,
            initial_alpha=1.0,
            n_step=1,
            model={
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
            }
        )
        .resources(num_gpus=0)
        .rollouts(num_rollout_workers=2)
        .framework("torch")
    )
    
    algo = config.build()
    
    logger.info("开始SAC训练...")
    for i in range(500):
        result = algo.train()
        
        episode_reward_mean = result.get("episode_reward_mean", 0)
        logger.info(f"Iteration {i+1}: Reward mean = {episode_reward_mean:.2f}")
        
        if (i + 1) % 50 == 0:
            checkpoint_path = algo.save(f"./checkpoints/sac_checkpoint_{i+1}")
            logger.info(f"检查点保存到: {checkpoint_path}")
    
    final_checkpoint = algo.save("./checkpoints/sac_final")
    logger.info(f"最终模型保存到: {final_checkpoint}")
    
    ray.shutdown()
    return algo

def evaluate_policy(algo, num_episodes=10):
    """评估训练好的策略"""
    env = make_close_combat_env()
    
    total_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        steps = 0
        
        while not (done or truncated):
            # 使用算法计算动作
            action = algo.compute_single_action(obs)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
        
        total_rewards.append(episode_reward)
        episode_lengths.append(steps)
        
        logger.info(f"Episode {episode+1}: "
                   f"Reward = {episode_reward:.2f}, "
                   f"Length = {steps}, "
                   f"Winner = {info.get('winner', 'N/A')}")
    
    env.close()
    
    avg_reward = np.mean(total_rewards)
    avg_length = np.mean(episode_lengths)
    
    logger.info(f"评估结果: "
               f"平均奖励 = {avg_reward:.2f}, "
               f"平均长度 = {avg_length:.2f}")
    
    return avg_reward, avg_length

def test_env():
    """测试环境基本功能"""
    logger.info("测试环境...")
    env = make_close_combat_env()
    
    obs, info = env.reset()
    logger.info(f"初始观察形状: {obs.shape}")
    logger.info(f"初始信息: {info}")
    
    # 测试随机动作
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        logger.info(f"Step {i+1}: "
                   f"Reward = {reward:.2f}, "
                   f"Done = {done}, "
                   f"Truncated = {truncated}")
        
        if done or truncated:
            logger.info(f"Episode结束: {info}")
            obs, info = env.reset()
    
    env.close()
    logger.info("环境测试完成")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ray RLlib训练脚本")
    parser.add_argument("--algorithm", type=str, default="ppo", 
                       choices=["ppo", "sac", "test"],
                       help="训练算法或测试环境")
    parser.add_argument("--eval", action="store_true",
                       help="评估现有模型")
    parser.add_argument("--checkpoint", type=str,
                       help="检查点路径用于评估")
    
    args = parser.parse_args()
    
    if args.algorithm == "test":
        test_env()
    elif args.eval:
        if not args.checkpoint:
            logger.error("评估需要指定检查点路径")
        else:
            # 加载模型进行评估
            ray.init(ignore_reinit_error=True)
            if "ppo" in args.checkpoint:
                config = PPOConfig().environment(env="CloseCombatEnv")
                algo = config.build()
            elif "sac" in args.checkpoint:
                config = SACConfig().environment(env="CloseCombatEnv")
                algo = config.build()
            else:
                config = PPOConfig().environment(env="CloseCombatEnv")
                algo = config.build()
            
            algo.restore(args.checkpoint)
            evaluate_policy(algo)
            ray.shutdown()
    else:
        if args.algorithm == "ppo":
            train_ppo()
        elif args.algorithm == "sac":
            train_sac()