from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
import pickle
import numpy as np
import plotly.graph_objects as go
from plotly.express.colors import sample_colorscale
import copy as cp

from diffusion_policy.common.viz_util import make_quiver, make_arrow

data = []
all_forces = []
class DiffusionUnetLowdimPolicy(BaseLowdimPolicy):
    def __init__(self, 
            model: ConditionalUnet1D,
            noise_scheduler: DDPMScheduler,
            horizon, 
            obs_dim, 
            action_dim, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_local_cond=False,
            obs_as_global_cond=False,
            pred_action_steps_only=False,
            oa_step_convention=False,
            # parameters passed to step
            **kwargs):
        super().__init__()
        assert not (obs_as_local_cond and obs_as_global_cond)
        if pred_action_steps_only:
            assert obs_as_global_cond
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_local_cond or obs_as_global_cond) else obs_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_local_cond = obs_as_local_cond
        self.obs_as_global_cond = obs_as_global_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.oa_step_convention = oa_step_convention
        self.kwargs = kwargs
        self.rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')

        self.force_adjust = True
        self.k_action_samples = 5
        self.min_force_threshold=10

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    def undo_transform_action(self, action):
        raw_shape = action.shape
        if raw_shape[-1] == 20:
            # dual arm
            action = action.reshape(-1,2,10)

        d_rot = action.shape[-1] - 4
        pos = action[...,:3]
        rot = action[...,3:3+d_rot]
        gripper = action[...,[-1]]
        rot = self.rotation_transformer.inverse(rot)
        uaction = np.concatenate([
            pos, rot, gripper
        ], axis=-1)

        if raw_shape[-1] == 20:
            # dual arm
            uaction = uaction.reshape(*raw_shape[:-1], 14)

        return uaction
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None, noisy_cond=False,
            # keyword arguments to scheduler.step
            **kwargs
            ):

        if noisy_cond:
            noise = .05
        else:
            noise = 0.0
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)

        condition_noise = torch.randn(
            size=global_cond.shape, 
            dtype=global_cond.dtype,
            device=global_cond.device,
            generator=generator) * noise
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        fig = go.Figure()

        def plot(traj):
            naction_pred = traj[...,:self.action_dim]
            action_pred = self.normalizer['action'].unnormalize(naction_pred)
            To = self.n_obs_steps
            start = To
            if self.oa_step_convention:
                start = To - 1
                end = start + self.n_action_steps
                action = action_pred[:,start:end]
            fig.add_trace(go.Scatter3d(x=action[0][:, 0], y=action[0][:, 1], z=action[0][:, 2]))

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]
            # if t%10 == 0:
            #     plot(trajectory)

            
            #condition_noise += class_guidance * 1


            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)#+condition_noise)

            class_guidance = torch.from_numpy(np.array([0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])) # torch.dot(torch.sum(trajectory[0][:, :3], axis=0), torch.from_numpy(np.array([0.0,0.0,1.0])).float())
            old_model_output = cp.deepcopy(model_output)
            model_output += class_guidance #* 5 # 10
            #print(model_output - old_model_output)
            #input("?")

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
            
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask] 
        # plot(trajectory)

        # fig.show()
        # input("?")

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        global data
        global all_forces
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """

        # for param in ['offset', 'scale']:  

        #     self.normalizer['obs'].params_dict[param] =  self.normalizer['obs'].params_dict[param][:-3]
        # for param in [max, mea]

        assert 'obs' in obs_dict
        assert 'past_action' not in obs_dict # not implemented yet

        #self.normalizer['obs'].params_dict['_default']['scale'] = self.normalizer['obs'].params_dict['_default']['scale']-3
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        B, _, Do = nobs.shape
        To = self.n_obs_steps
        
        assert Do == self.obs_dim
        T = self.horizon
        Da = self.action_dim

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_local_cond:
            # condition through local feature
            # all zero except first To timesteps
            local_cond = torch.zeros(size=(B,T,Do), device=device, dtype=dtype)
            local_cond[:,:To] = nobs[:,:To]
            shape = (B, T, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        elif self.obs_as_global_cond:
            # condition throught global feature
            global_cond = nobs[:,:To].reshape(nobs.shape[0], -1)
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            shape = (B, T, Da+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs[:,:To]
            cond_mask[:,:To,Da:] = True

        #fig = go.Figure()
        colors = ['blues', 'greens', 'oranges', 'greys', 'purples', 'reds']
        scores = []
        actions = []

        all_forces.append(obs_dict['force_info'])
        force_adjust_vals = np.where(abs(obs_dict['force_info'][:, -1, :3]) > 10, \
                                                 obs_dict['force_info'][:, -1, :3], \
                                                 np.zeros_like(obs_dict['force_info'][:, -1, :3]))
        force_adjust_vals = obs_dict['force_info'][:, -1, :3].numpy()
        force_adjust_vals = np.zeros_like(force_adjust_vals)
        force_adjust_vals[:, 2] = np.ones_like(force_adjust_vals[:, 2])
        # print(force_adjust_vals)
        # input("?")

        for i in range(self.k_action_samples):
            # run sampling
            if self.force_adjust and np.any(force_adjust_vals != 0):
                nsample = self.conditional_sample(
                    cond_data, 
                    cond_mask,
                    local_cond=local_cond,
                    global_cond=global_cond,
                    noisy_cond=False,
                    **self.kwargs)

                # nsample = self.conditional_sample(
                #     cond_data, 
                #     cond_mask,
                #     local_cond=local_cond,
                #     global_cond=global_cond,
                #     noisy_cond=True,
                #     **self.kwargs)
            else:
                nsample = self.conditional_sample(
                    cond_data, 
                    cond_mask,
                    local_cond=local_cond,
                    global_cond=global_cond,
                    noisy_cond=False,
                    **self.kwargs)

            # unnormalize prediction
            naction_pred = nsample[...,:Da]
            action_pred = self.normalizer['action'].unnormalize(naction_pred)

            # get action
            if self.pred_action_steps_only:
                action = action_pred
            else:
                start = To
                if self.oa_step_convention:
                    start = To - 1
                end = start + self.n_action_steps
                action = action_pred[:,start:end]

            if self.force_adjust and np.any(force_adjust_vals != 0):
                if not(np.isnan(action).any() or np.isinf(action).any()):
                    #instead just try getting the dot product of the vectors
                    #truncate if the force is less than 10? 
                    #pick the one with the smallest dot
                    #make sure score is nan for nan moves
                    rel_action = self.undo_transform_action(action)[:, :, :6]
                    dot_products = [np.dot(force_adjust_vals[0], rel_action[:, j, :3][0]) for j in range(rel_action.shape[1])]

                    #adjustment_thresh_mask = abs(obs_dict['force_info'][:, -1, :]) > self.min_force_threshold
                    #adjustment_direction = obs_dict['force_info'][:, -1, :] > 0
                    #rel_action_direction = rel_action > 0
                    #rel_action_mask = (adjustment_direction & rel_action_direction) | (np.logical_not(adjustment_direction) & np.logical_not(rel_action_direction))
                    #rel_action_mask = rel_action_mask & adjustment_thresh_mask

                    #direction_masked_rel_action = np.where(rel_action_mask, rel_action, np.zeros_like(rel_action))
                    scores.append(np.sum(dot_products))


                    # print(obs_dict['force_info'])
                    # print(force_adjust_vals)
                    # print(rel_action)
                    # print(dot_products)
                    # print(scores)
                    # input("Next?")
                    
                    actions.append(action)
            else:
                if not(np.isnan(action).any() or np.isinf(action).any()):
                    print(action)
                    print(np.isnan(action))
                    print(np.isinf(action))
                    break
            

        if self.force_adjust and np.any(force_adjust_vals != 0):

            action = actions[np.nanargmax(scores)]
            try:
                colors = sample_colorscale('Viridis', (scores - np.nanmin(scores))/np.nanmax((scores - np.nanmin(scores))))
            except ValueError:
                print(actions)
                print((scores - np.nanmin(scores))/np.nanmax((scores - np.nanmin(scores))))

            start_position = obs_dict['obs'][0, -1, 15:18]#.reshape(1, -1)

            
            i = 0
            # print((scores - np.nanmin(scores))/np.nanmax((scores - np.nanmin(scores))))
            # print(colors)
            for action_traj in actions:
                quiver_starts = []
                quiver_ends = []
                # print(action_traj[-1])
                # input("?")
                for step in range(len(action_traj[-1])-1):
                    quiver_starts.append(action_traj[-1][step])
                    quiver_ends.append(action_traj[-1][step + 1])

                # print(quiver_starts)
                # print(quiver_ends)
                # input("?")
                data += make_quiver(quiver_starts, quiver_ends, [colors[i] for _ in range(len(quiver_starts))])
                i += 1
        else: 
            colors = sample_colorscale('Viridis', [0])
            quiver_starts = []
            quiver_ends = []
            for step in range(len(action[-1])-1):
                    quiver_starts.append(action[-1][step])
                    quiver_ends.append(action[-1][step + 1])
            data += make_quiver(quiver_starts, quiver_ends, [colors[0] for _ in range(len(quiver_starts))])



        pickle.dump(data, open('policy_action_data.pkl', 'wb'))
        pickle.dump(all_forces, open('policy_forces.pkl', 'wb'))
        result = {
            'action': action,
            'action_pred': action_pred
        }
        if not (self.obs_as_local_cond or self.obs_as_global_cond):
            nobs_pred = nsample[...,Da:]
            obs_pred = self.normalizer['obs'].unnormalize(nobs_pred)
            action_obs_pred = obs_pred[:,start:end]
            result['action_obs_pred'] = action_obs_pred
            result['obs_pred'] = obs_pred
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())


    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch['obs']
        action = nbatch['action']

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = action
        if self.obs_as_local_cond:
            # zero out observations after n_obs_steps
            local_cond = obs
            local_cond[:,self.n_obs_steps:,:] = 0
        elif self.obs_as_global_cond:
            global_cond = obs[:,:self.n_obs_steps,:].reshape(
                obs.shape[0], -1)
            if self.pred_action_steps_only:
                To = self.n_obs_steps
                start = To
                if self.oa_step_convention:
                    start = To - 1
                end = start + self.n_action_steps
                trajectory = action[:,start:end]
        else:
            trajectory = torch.cat([action, obs], dim=-1)

        # generate impainting mask
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = trajectory[condition_mask]
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss
