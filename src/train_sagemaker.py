import os
import argparse
import torch
from sagemaker_training import environment
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CallbackList
from wandb.integration.sb3 import WandbCallback
from DTUWindGym.envs import WindFarmEnv
from py_wake.examples.data.hornsrev1 import V80 as wind_turbine

# Local imports
from utils import (
    WindFarmPolicy,
    WindFarmMonitor,
    CurriculumCallback,
    make_env,
    parse_args
)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else 
                     "mps" if torch.backends.mps.is_available() else 
                     "cpu")

print(f"Using device: {device}")

if __name__ == '__main__':

    args = parse_args()
    #curriculum_steps = 200
    curriculum_steps = args.train_steps // 2  
    pure_similarity_steps = args.train_steps // 4  

    wandb.init(project="WindFarm_Curriculum", 
               name=f"curriculum_dt{args.dt_env}_pow{args.power_avg}_seed{args.seed}",
               config=vars(args))


    wandb.run.summary.update({
        "views/training_progress": {
            "version": "v2",
            "panels": [
                {
                    "panel_type": "line",
                    "title": f"Yaw Angles - Env {i}",
                    "source": "glob",
                    "series": [f"states/yaw_angle_t1-{i}", f"states/yaw_angle_t2-{i}"]
                } for i in range(args.n_env)
            ] + [
                {
                    "panel_type": "line",
                    "title": f"Power - Env {i}",
                    "source": "glob",
                    "series": [f"charts/agent_power-{i}", f"charts/base_power-{i}"]
                } for i in range(args.n_env)
            ]
        }
    })

    # fake env for processing purposes
    temp_env = WindFarmEnv(
        turbine=wind_turbine(),
        yaml_path=args.yaml_path,
        TurbBox=args.turbbox_path,
        seed=args.seed,
        dt_env=args.dt_env,
        observation_window_size=args.lookback_window
    )
    n_features = temp_env._get_num_raw_features()

    # real env
    env = DummyVecEnv([make_env(args.seed, args.yaml_path, args.turbbox_path, 
                                args.dt_env, args.power_avg, curriculum_steps, pure_similarity_steps) for _ in range(args.n_env)])

    model = PPO(WindFarmPolicy, env, 
                n_steps=512,
                batch_size=256,
                n_epochs=15,
                ent_coef=args.ent_coef, 
                learning_rate=args.learning_rate, 
                #clip_range=0.2,
                policy_kwargs={
                    "n_features": temp_env._get_num_raw_features(),
                    "window_size": args.lookback_window
                    },
                verbose=1, 
                device=device)

    callbacks = CallbackList([
        WandbCallback(gradient_save_freq=0, verbose=2, model_save_path='model', model_save_freq=10000),
        WindFarmMonitor(),
        CurriculumCallback(args.train_steps)
    ])

    model.learn(total_timesteps=args.train_steps, callback=callbacks)
    model.save("windfarm_curriculum_model")
