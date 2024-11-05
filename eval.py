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
from diffusion_policy.workspace.base_workspace import BaseWorkspace

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
@click.option('-d', '--device', default='cuda:0')

def main(checkpoint, output_dir, device):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    # cfg['obs_dim'] = 20

    # cfg.policy['model']['global_cond_dim'] = 40

    cfg['task']['obs_keys'] += ['robot0_ee_force']
    cfg['task']['obs_keys'] += ['robot0_ee_torque']
    cfg['task']['env_runner']['obs_keys'] += ['robot0_ee_force']
    cfg['task']['env_runner']['obs_keys'] += ['robot0_ee_torque']

    print(cfg['task']['obs_keys'])

    # pretty(cfg)
    # exit(0)
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)#, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model
    # print(type(policy))
    # print(policy.kwargs)
    # exit(0)
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    # device = torch.device(device)
    # policy.to(device)
    # policy.eval()
    # print(cfg)
    # print()
    # print(cfg.task)
    # exit(0)
    cfg.task.env_runner['n_train'] = 0#1 
    cfg.task.env_runner['n_train_vis'] = 0#1 
    cfg.task.env_runner['n_test'] = 1  
    cfg.task.env_runner['n_envs'] = 1
    cfg.task.env_runner['n_test_vis'] = 1

    #set whether to do the force thing

    # run eval
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir)
    runner_log = env_runner.run(policy)
    
    # dump log to json
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value
    out_path = os.path.join(output_dir, 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

if __name__ == '__main__':
    main()
