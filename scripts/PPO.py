import warnings
import torch
from omnisafe.common.experiment_grid import ExperimentGrid
from omnisafe.utils.exp_grid_tools import train

if __name__ == '__main__':
    eg = ExperimentGrid(exp_name='PPO-Safety-Benchmark')
    
    # Algorithm and Env
    eg.add('algo', ['PPO'])
    eg.add('env_id', ['SafetyPointGoal1-v0'])
    eg.add('seed', [0, 1, 2])

    # 1. Training Configs (1 Million steps = 500 Epochs)
    eg.add('train_cfgs:total_steps', [1024000]) 
    eg.add('train_cfgs:vector_env_nums', [1])
    eg.add('train_cfgs:parallel', [1])
    eg.add('train_cfgs:device', ['cuda:0' if torch.cuda.is_available() else 'cpu'])

    # 2. Algorithm Configs
    eg.add('algo_cfgs:steps_per_epoch', [2048])
    eg.add('algo_cfgs:update_iters', [10])
    
    # NOTE: No lagrange_cfgs or cost_limit here! 
    # Standard PPO is unconstrained by definition.

    # 3. Logger Configs
    eg.add('logger_cfgs:use_wandb', [False])
    eg.add('logger_cfgs:use_tensorboard', [True])
    eg.add('logger_cfgs:save_model_freq', [50])

    available_gpus = list(range(torch.cuda.device_count()))
    gpu_id = [0] if available_gpus else None

    print("Launching standard PPO Grid (Unconstrained Baseline)...")
    eg.run(train, num_pool=3, gpu_id=gpu_id)