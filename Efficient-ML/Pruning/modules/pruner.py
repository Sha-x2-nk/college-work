import math

import jax
from jax import numpy as jnp

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from modules.trainer import TrainerModule

def sensitivity_scan(tm: TrainerModule, test_data, scan_step = 0.1, scan_start = 0.1, scan_stop = 1.0, verbose = True):
  sparsities = jnp.arange(start = scan_start, stop = scan_stop, step = scan_step)
  accuracies = dict()
  model_params = tm.state.params
  layer_names = list(model_params.keys())

  for layer in layer_names:
    param = model_params[layer]['kernel']
    param_copy = param.copy()

    accuracy = []

    for sparsity in sparsities:
      mask = FineGrainedPruner.pruner_param(param, sparsity)
      param *= mask
      model_params[layer]['kernel'] = param
      acc = tm.test(test_data)
      accuracy.append(acc)
      model_params[layer]['kernel'] = param_copy

    accuracies[layer] = accuracy
    
    if verbose:
      print(f"layer: {layer} | Acc: [{','.join([f'{{:.2f}}'.format(x) for x in accuracy])}] | sparsity: [{', '.join([f'{{:.2f}}'.format(x) for x in sparsities])}]")

  return accuracies, sparsities

def plot_sensitivity_scan(sparsities, accuracies: dict):
  # Find the minimum and maximum accuracy values across all layers
  min_accuracy = min([min(acc) for acc in accuracies.values()])
  max_accuracy = max([max(acc) for acc in accuracies.values()])

  # Adjust the range to fit your data, adding a small margin
  y_axis_range = [math.floor(min_accuracy) - 0.05, math.ceil(max_accuracy) + 0.05]

  layer_names = list(accuracies.keys())
  fig = make_subplots(rows=2, cols=2, subplot_titles=layer_names)

  for plot_idx, layer in enumerate(layer_names):
    # Get accuracies for the current plot
    current_accuracies = accuracies[layer]
    row, col = divmod(plot_idx, 2)
      
    # Add scatter plot for accuracy vs sparsity
    fig.add_trace(go.Scatter(
      x=sparsities,
      y=current_accuracies,
      mode='lines+markers',
      name=f'Layer {layer} Accuracy',
      line=dict(color='blue')),
      row=row+1, col=col+1
    )
      
    # Set x and y axis labels, ticks, and limits
    fig.update_xaxes(
      title_text="Sparsity", 
      range=[0.1, 1.0], 
      tickvals=jnp.arange(0, 1.0, 0.1), 
      row=row+1, col=col+1
    )
    fig.update_yaxes(title_text="Top-1 Accuracy", range=y_axis_range, row=row+1, col=col+1)


    # Update the title and layout
    fig.update_layout(
      height=800, 
      width=1000, 
      title_text='Sensitivity Curves: Validation Accuracy vs. Pruning Sparsity',
      showlegend=False
    )

  fig.show()

class FineGrainedPruner:
  def __init__(self, tm: TrainerModule, sparsity_dict: dict):
    self.tm = tm
    self.masks = FineGrainedPruner.pruner(self.tm.state.params, sparsity_dict)

  def apply(self):
    layer_names = list(self.tm.state.params.keys())

    for layer in layer_names:
      # We will only prune the weights
      self.tm.state.params[layer]['kernel'] *= self.masks[layer]
  
  @staticmethod
  def pruner_param(param: jax.Array, sparsity: float):
    sparsity = min(max(0.0, sparsity), 1.0)
    if sparsity == 1.0:
      return jnp.zeros_like(param)
    elif sparsity == 0.0:
      return jnp.ones_like(param)
    
    threshold = jnp.percentile(jnp.abs(param.flatten()), sparsity * 100)
    mask = jnp.abs(param) > threshold
    return mask
    
  @staticmethod
  def pruner(model_params: dict, sparsity_dict: dict):
    masks = dict()
    layer_names = list(model_params.keys())

    for layer in layer_names:
      # We will only prune the weights
      param = model_params[layer]['kernel']
      masks[layer] = FineGrainedPruner.pruner_param(param, sparsity_dict[layer])

    return masks
  
class ChannelPruner:
  def __init__(self, tm: TrainerModule):
    self.tm = tm

  def get_num_channels_to_keep(self, num_channels: int, prune_ratio: float) -> int:
    return int(num_channels * (1 - prune_ratio))

  # Prune all conv layers by the same ratio
  # FLAX stores filters as F, F, IN_CHANNEL, OUT_CHANNEL
  def apply(self, prune_ratio: float):
    layer_names = list(self.tm.state.params.keys())

    for layer_idx, layer_name in enumerate(layer_names):
      if 'Conv' in layer_name:
        # Decrease the output channels
        org_out_channels = self.tm.state.params[layer_name]['kernel'].shape[-1]
        new_out_channels = self.get_num_channels_to_keep(org_out_channels, prune_ratio)

        self.tm.model.out_channels[layer_name] = new_out_channels

        self.tm.state.params[layer_name]['kernel'] = self.tm.state.params[layer_name]['kernel'][:, :, :, :new_out_channels]
        self.tm.state.params[layer_name]['bias'] = self.tm.state.params[layer_name]['bias'][:new_out_channels]

        if layer_idx + 1 < len(layer_names):
          next_layer_idx = layer_idx + 1
        
          next_layer_name = layer_names[next_layer_idx]
          # next layer is conv, reduce in_channels
          if 'Conv' in next_layer_name:
            self.tm.state.params[next_layer_name]['kernel'] = self.tm.state.params[next_layer_name]['kernel'][:, :, :new_out_channels, :]

          # next layer is dense
          elif 'Dense' in next_layer_name:
            old_conv_output = self.tm.state.params[next_layer_name]['kernel'].shape[0]
            new_conv_output = int((old_conv_output / org_out_channels) * new_out_channels)
            self.tm.state.params[next_layer_name]['kernel'] = self.tm.state.params[next_layer_name]['kernel'][:new_conv_output,:]
