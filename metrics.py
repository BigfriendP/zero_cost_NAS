import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import types

"""# Score functions definition"""

"""## Standard score function"""

def compute_score_metric(network, inputs, batch_size):

  def counting_forward_hook(module, inp, out):
    try:
      if isinstance(inp, tuple):
        inp = inp[0]
      inp = inp.view(inp.size(0), -1)
      x = (inp > 0).float()
      K = x @ x.t()
      K2 = (1.-x) @ (1.-x.t())
      network.K = network.K + K.cpu().numpy() + K2.cpu().numpy()
    except:
      pass

  network.K = np.zeros((batch_size, batch_size))

  for name, module in network.named_modules():
    if 'ReLU' in str(type(module)):
      module.register_forward_hook(counting_forward_hook)
  
  with torch.no_grad():
    y, out = network(inputs)

  
  y = y.detach()
  out = out.detach()
  inputs.detach()
  
  sign, score = np.linalg.slogdet(network.K)


  return score

"""## Snip score function"""

def sum_arr(arr):
  sum = 0.
  for i in range(len(arr)):
    sum += torch.sum(arr[i])
  return sum.item()

def snip_forward_conv2d(self, x):
  return F.conv2d(x, self.weight * self.weight_mask, self.bias, 
                  self.stride, self.padding, self.dilation, self.groups)

def snip_forward_linear(self, x):
  return F.linear(x, self.weight * self.weight_mask, self.bias)

def compute_snip_score(network, inputs, targets, loss_fn):
  
  for layer in network.modules():
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
      layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
      layer.weight.requires_grad = False

      # Override the forward methods:
      if isinstance(layer, nn.Conv2d):
        layer.forward = types.MethodType(snip_forward_conv2d, layer)

      if isinstance(layer, nn.Linear):
        layer.forward = types.MethodType(snip_forward_linear, layer)

  # Compute gradients (but don't apply them)
  network.zero_grad()

  outputs = network.forward(inputs)
  loss = loss_fn(outputs[1], targets)
  loss.backward()


  # select the gradients that we want to use for search/prune
  def snip(layer):
    if layer.weight_mask.grad is not None:
      return torch.abs(layer.weight_mask.grad)
    else:
      return torch.zeros_like(layer.weight)

  metric_array = []

  for layer in network.modules():
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
      metric_array.append(snip(layer))
  
  snip_score = sum_arr(metric_array)
  return snip_score

"""## Synflow score function"""

def no_op(self,x):
  return x

def compute_synflow_score(network, inputs, bn=False, loss_fn=None):

  device = inputs.device
  
  if bn == False:
    for module in network.modules():
      if isinstance(module,nn.BatchNorm2d) or isinstance(module,nn.BatchNorm1d) :
        module.forward = types.MethodType(no_op, module)


  #convert params to their abs. Keep sign for converting it back.
  @torch.no_grad()
  def linearize(network):
    signs = {}
    for name, param in network.state_dict().items():
      signs[name] = torch.sign(param)
      param.abs_()
    return signs

  #convert to orig values
  @torch.no_grad()
  def nonlinearize(network, signs):
    for name, param in network.state_dict().items():
      if 'weight_mask' not in name:
        param.mul_(signs[name])

  # keep signs of all params
  signs = linearize(network)
    
  # Compute gradients with input of 1s 
  network.zero_grad()
  network.double()
  input_dim = list(inputs[0,:].shape)
  inputs = torch.ones([1] + input_dim).double().to(device)
  output = network.forward(inputs)
  torch.sum(output[1]).backward() 

  # select the gradients that we want to use for search/prune
  def synflow(layer):
    if layer.weight.grad is not None:
      return torch.abs(layer.weight * layer.weight.grad)
    else:
      return torch.zeros_like(layer.weight)

  metric_array = []

  for layer in network.modules():
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
      metric_array.append(synflow(layer))
  
  # apply signs of all params
  nonlinearize(network, signs)

  synflow_score = sum_arr(metric_array)
  return synflow_score


