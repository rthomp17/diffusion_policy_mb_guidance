import torch 
import json
import numpy as np
from abc import ABC, abstractmethod

class BaseGuideFunction(ABC):
    def __init__(self, normalizer, activation=True, weight=1, ):
        self.normalizer=normalizer
        self.activation=activation
        self.weight=weight

    @abstractmethod
    def guide_gradient(self, action_sequence):
        pass

class BaseActivationFunction(ABC):
    def __init__(self, normalizer):
        self.normalizer=normalizer

    @abstractmethod
    def check_active(self, action_sequence):
        pass


class within_activation(BaseActivationFunction):
    def __init__(self, normalizer, x_range=[-10, 10], y_range=[-10, 10], z_range=[-10, 10]):
        super().__init__(normalizer)
        self.bb_low = self.normalizer.normalize(torch.Tensor([x_range[0], y_range[0], z_range[0]] + [0,0,0,0,0,0,0]))
        self.bb_high = self.normalizer.normalize(torch.Tensor([x_range[1], y_range[1], z_range[1]] + [0,0,0,0,0,0,0]))

    def check_active(self, action_sequence):
        return action_sequence[-1, :, :3] > self.bb_low and action_sequence[-1, :, :3] < self.bb_high 
    
class distance_thresh_activation(BaseActivationFunction):
    def __init__(self, normalizer, center=[0,0,0], dist=10):
        super().__init__(normalizer)
        self.center = self.normalizer['action'].normalize(torch.Tensor(center + [0,0,0,0,0,0,0]))[:3]
        self.dist = dist

    def check_active(self, action_sequence):
        return np.linalg.norm(action_sequence[-1, :, :3] - self.center) < self.dist

class in_direction(BaseGuideFunction):
    def __init__(self, normalizer, weight=1, activation=True, direction=[0,0,1]):
        super().__init__(normalizer, activation, weight)
        self.direction = torch.Tensor(direction)

    def guide_gradient(self, action_sequence):
        guidance = torch.zeros_like(action_sequence)
        if self.activation:
            for i in range(guidance.shape[1]):
                guidance[:, i, :3] = self.direction

        return guidance * self.weight
    
class repel_point(BaseGuideFunction):
    def __init__(self, normalizer, activation=True, weight=1, point=[0,0,0]):
        super().__init__(normalizer, activation, weight)
        #TODO clean up to make the normalizer just work for ee pose
        self.point = self.normalizer['action'].normalize(torch.Tensor(point + [0,0,0,0,0,0,0]))[:3]

    def guide_gradient(self, action_sequence):
        guidance = torch.zeros_like(action_sequence)
        if self.activation: 
            target = torch.zeros_like(action_sequence)

            for i in range(target.shape[1]):
                target[:, i, :3] = self.point
                target[:, i, 3:] = action_sequence[:, i, 3:]

            with torch.enable_grad():
                naction = action_sequence.clone().detach().requires_grad_(True)
                dist = torch.linalg.norm(naction - target, dim=2, ord=2)# (B, pred_horizon)
                dist = 1/dist.mean(dim=1) # (B,)
                grad = torch.autograd.grad(dist, naction, grad_outputs=torch.ones_like(dist), create_graph=False)[0]
                guidance = grad
        return guidance * self.weight

class attract_point(BaseGuideFunction):
    def __init__(self, normalizer, activation, weight=1, point=[0,0,0]):
        super().__init__(normalizer, activation, weight)
        #TODO clean up to make the normalizer just work for ee pose
        self.point = self.normalizer['action'].normalize(torch.Tensor(point + [0,0,0,0,0,0,0]))[:3]

    #attraction weakens at a distance and increases closer to the point focus
    def guide_gradient(self, action_sequence):
        guidance = torch.zeros_like(action_sequence)
        if self.activation: 
            target = torch.zeros_like(action_sequence)

            for i in range(target.shape[1]):
                target[:, i, :3] = self.point
                target[:, i, 3:] = action_sequence[:, i, 3:]

            with torch.enable_grad():
                naction = action_sequence.clone().detach().requires_grad_(True)
                dist = torch.linalg.norm(naction - target, dim=2, ord=2) # (B, pred_horizon)
                dist = 1/dist.mean(dim=1) # (B,)
                grad = -torch.autograd.grad(dist, naction, grad_outputs=torch.ones_like(dist), create_graph=False)[0]
                guidance = grad
        return guidance * self.weight
    
def guide_factory(json_file, normalizer):
    with open(json_file) as f:
        component_functions = json.load(f)

    components = []
    for component in component_functions:
        if component["activation"] == "null":
            activation = True
        else:
            activation = eval(component["activation"])(**component["activation_args"])
        components.append(eval(component["function"])(normalizer=normalizer,
                                                      activation=activation,
                                                      weight=component["weight"],
                                                      **component["function_args"]))
        
    return lambda action_sequence: sum([component.guide_gradient(action_sequence) for component in components])
                
            
            
