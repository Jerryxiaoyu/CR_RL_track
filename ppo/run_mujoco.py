#!/usr/bin/env python3
import argparse
from baselines.common.cmd_util import mujoco_arg_parser
from baselines import bench, logger

from my_envs.mujoco import *
import tensorflow as tf
import os
import time
import datetime
dir = os.path.join(os.getcwd(),'log-files',
                   datetime.datetime.now().strftime("ppo-%Y-%m-%d-%H-%M-%S-%f"))

def argsparser():
    parser = argparse.ArgumentParser("Tensorflow Implementation of MAML")
    
    parser.add_argument('--env', help='environment ID', type=str, default='HalfCheetahTrack-v2')  # Reacher-v2
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(5e6))
    
    args = parser.parse_args()
    print(args)
    return args

def train(env_id, num_timesteps, seed):
    from baselines.common import set_global_seeds
    from baselines.common.vec_env.vec_normalize import VecNormalize
    from baselines.ppo2 import ppo2
    from baselines.ppo2.policies import MlpPolicy
    import gym
    import tensorflow as tf
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
    ncpu = 1
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    config.gpu_options.per_process_gpu_memory_fraction = 1 / 2.
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)
    sess.__enter__()
    def make_env():
        env = gym.make(env_id)
        env = bench.Monitor(env, logger.get_dir())
        return env
    env = DummyVecEnv([make_env])
    env = VecNormalize(env)

    set_global_seeds(seed)
    policy = MlpPolicy
    ppo2.learn(policy=policy, env=env, nsteps=2048*10, nminibatches=32*10,
        lam=0.95, gamma=0.99, noptepochs=10, log_interval=1,
        ent_coef=0.0,
        lr=3e-4,
        cliprange=0.2,
        total_timesteps=num_timesteps)

    Saver = tf.train.Saver(max_to_keep=10)
    Saver.save(sess, os.path.join(dir,  'trained_variables.ckpt'), write_meta_graph=False)

def main():
    args = argsparser()
    logger.configure(dir)
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)

if __name__ == '__main__':
    main()
