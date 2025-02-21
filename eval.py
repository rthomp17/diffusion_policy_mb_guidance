"""
Usage:
python eval.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
import pickle
import numpy as np
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.common.guidance_util import guide_factory

def pretty(d, indent=0):
   for key, value in d.items():
      print('\t' * indent + str(key))
      if isinstance(value, dict):
         pretty(value, indent+1)
      else:
         print('\t' * (indent+1) + str(value))

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-g', '--guidance_config', required=True)
@click.option('-n', '--num_test', default=1)
@click.option('-s', '--stochastic_sampling', is_flag=True)
@click.option('-d', '--device', default='cuda:0')

def main(checkpoint, output_dir, guidance_config, num_test, stochastic_sampling, device,):
    # if os.path.exists(output_dir):
    #     click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    print( payload['cfg']['task'])
    # cfg['obs_dim'] = 20

    # cfg.policy['model']['global_cond_dim'] = 40
    cfg['task']['obs_keys'] += ['robot0_ee_force']
    cfg['task']['obs_keys'] += ['robot0_ee_torque']
    cfg['task']['env_runner']['obs_keys'] += ['robot0_ee_force']
    cfg['task']['env_runner']['obs_keys'] += ['robot0_ee_torque']

    # pretty(cfg)
    # exit(0)
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)#, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model
    policy.stochastic_sampling = stochastic_sampling
    
    # print(type(policy))
    # print(policy.kwargs)
    # exit(0)
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    device = torch.device(device)
    policy.to(device)
    # policy.eval()
    # print(cfg)
    # print()
    # print(cfg.task)
    # exit(0)
    cfg.task.env_runner['n_train'] = 0#1 
    cfg.task.env_runner['n_train_vis'] = 0#1 
    cfg.task.env_runner['n_test'] = num_test  
    cfg.task.env_runner['n_envs'] = num_test
    cfg.task.env_runner['n_test_vis'] = num_test


    # run eval
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir)
    
    guidance = guide_factory(guidance_config, policy.normalizer)
    runner_log = env_runner.run(policy, guidance=guidance)
    
    # dump log to json
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        elif isinstance(value, list):
            print(output_dir)
            print(key)
            pickle.dump(value, open(os.path.join(output_dir, f'{key}'), 'wb'))
            json_log[key] = os.path.join(output_dir, 'action_histories', f'{key}')
        else:
            json_log[key] = value
    out_path = os.path.join(output_dir, 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

if __name__ == '__main__':
    main()
