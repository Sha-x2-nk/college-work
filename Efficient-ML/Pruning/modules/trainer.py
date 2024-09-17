import math

import jax
import jax.numpy as jnp

# FLAX
from flax import linen as nn
from flax.training import train_state

# Optax
import optax

# Plotly
from plotly.subplots import make_subplots
from plotly import graph_objects as go

class TrainerModule:
  # Constructor
  def __init__(self, model_class: nn.Module, model_hparams: dict, optimizer_name: str, learning_rate: float, exmp_imgs: jax.Array):
    self.model = model_class(**model_hparams)
    self.optimizer_name = optimizer_name
    self.learning_rate = learning_rate
    self.exmp_imgs = exmp_imgs
    self.init_model()
    self.init_optimizer()
    self.init_train_state(self.model.apply, self.init_params, self.tx)
    self.create_functions()

  # Init model parameters and optimiser
  def init_model(self, seed = 42):
    init_rng = jax.random.PRNGKey(seed)
    variables = self.model.init(init_rng, self.exmp_imgs)    
    self.init_params = variables['params']
    
  def init_optimizer(self):
    if self.optimizer_name.lower() == "adam":
      self.tx = optax.adam(learning_rate= self.learning_rate)
  
  def init_train_state(self, apply_fn, params, tx):
    self.state = train_state.TrainState.create(apply_fn=apply_fn, params= params, tx= tx)

  # creating jax JIT compiled train and eval steps
  def create_functions(self):
    def train_step(state, images, labels):
      def loss_fn(params):
        one_hot = jax.nn.one_hot(labels, 10)
        logits = state.apply_fn({'params': params}, images)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        return loss, logits
      
      grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
      (loss, logits), grads = grad_fn(state.params)
      accuracy = jnp.mean(jnp.argmax(logits, axis=1) == labels)
      state = state.apply_gradients(grads=grads)
      return state, loss, accuracy
    self.train_step = jax.jit(train_step)

    def eval_step(state, images, labels):
      logits = state.apply_fn({'params': state.params}, images)
      accuracy = jnp.mean( jnp.argmax(logits, 1) == labels )
      return accuracy
    self.eval_step = jax.jit(eval_step)

  # Train loop - In callback we will pass pruner.apply, in order to 0 out the masked elements
  def train(self, train_data, val_data, num_epochs = 20, callbacks = None, verbose = True):
    batch_size = 512
    train_images, train_labels = train_data
    total_iters = math.ceil(len(train_labels)/batch_size)
    train_images_loader = jnp.array_split(train_images, total_iters)
    train_labels_loader = jnp.array_split(train_labels, total_iters)

    val_images, val_labels = val_data

    best_state = None
    best_acc = {'val': 0, 'train': 0}

    for epoch in range(1, num_epochs + 1):
      for x_train,y_train in zip(train_images_loader, train_labels_loader):
        self.state, train_loss, train_acc = self.train_step(self.state, x_train, y_train)
        if callbacks is not None:
          for callback in callbacks:
            callback()

      val_acc = self.eval_step(self.state, val_images, val_labels)
      
      if best_acc['val'] < val_acc or (best_acc['val'] == val_acc and best_acc['train'] > train_acc):
        best_acc['val'] = val_acc
        best_acc['train'] = train_acc
        best_state = self.state
        if verbose:
          print("NEW BEST FOUND", end= " | ")

      if verbose:
        print(f"Epoch: {epoch} | train_loss: {train_loss} | train_acc: {train_acc} | val_acc: {val_acc}")

    self.state = best_state
    return best_state

  # Test
  def test(self, test_data) -> float:
    test_images, test_labels = test_data
    test_acc = self.eval_step(self.state, test_images, test_labels)
    return test_acc
  
  # Utility Function
  def get_model_sparsity(self) -> float:
    num_zeros = 0
    num_total = 0

    layer_names = list(self.state.params.keys())

    for layer in layer_names:
      for type in ['kernel', 'bias']:
        param = self.state.params[layer][type]

        num_zeros += (param == 0).sum()
        num_total += param.size

    return num_zeros / num_total
    
  # Utility Function
  def get_model_size(self) -> str:
    num_bytes = 0

    layer_names = list(self.state.params.keys())

    for layer in layer_names:
      for type in ['kernel', 'bias']:
        param = self.state.params[layer][type]
        
        num_bytes += (param != 0).sum() * (param.dtype.itemsize)

    num_bytes_kb = num_bytes / 1024
    num_bytes_mb = num_bytes_kb / 1024
    num_bytes_gb = num_bytes_mb / 1024

    if num_bytes_gb >= 1:
      return f"{num_bytes_gb:.2f} GiB"
    elif num_bytes_mb >= 1:
      return f"{num_bytes_mb:.2f} MiB"
    elif num_bytes_kb >= 1:
      return f"{num_bytes_kb:.2f} KiB"
    else:
      return f"{num_bytes:.2f} B"
    
  # Utility Function
  def plot_weight_distribution(self, bins=256, count_nonzero_only=False, color='blue', fig=None):
    layer_names = list(self.state.params.keys())
    
    if fig is None:
      fig = make_subplots(rows=2, cols=2, subplot_titles=layer_names)

    for plot_idx, layer in enumerate(layer_names):
      # Not counting bias
      param = self.state.params[layer]['kernel']
      param_flat = param.flatten()
      if count_nonzero_only:
          param_flat = param_flat[param_flat != 0]
      
      hist, bin_edges = jnp.histogram(param_flat, bins=bins, density=True)
      
      row, col = divmod(plot_idx, 2)
      fig.add_trace(go.Bar(
          x=bin_edges[:-1],
          y=hist,
          marker=dict(color=color, opacity=0.5),
          showlegend=False
      ), row=row+1, col=col+1)
      
      fig.update_xaxes(title_text=layer, row=row+1, col=col+1)
      fig.update_yaxes(title_text='Density', row=row+1, col=col+1)
            
    fig.update_layout(height=600, width=1000, title_text='Histogram of Weights')
    return fig

  # Utility Function
  def plot_num_parameters(self, count_nonzero_only=False, color='blue', fig=None):
    num_parameters = dict()

    layer_names = list(self.state.params.keys())
    for layer in layer_names:
        if count_nonzero_only:
          num_parameters[layer] = (self.state.params[layer]['kernel']!=0).sum()
        else:
          num_parameters[layer] = self.state.params[layer]['kernel'].size
    
    if fig is None:
      fig = go.Figure()

    fig.add_trace(go.Bar(
        x=list(num_parameters.keys()),
        y=list(num_parameters.values()),
        text=list(num_parameters.values()),
        textposition='auto',
        marker=dict(color=color)
    ))
    
    fig.update_layout(
        title='#Parameter Distribution',
        xaxis_title='Layer Name',
        yaxis_title='Number of Parameters',
        xaxis_tickangle=-60,
        showlegend=False,
        bargap=0.2,
        plot_bgcolor='white',
        height=600,
        width=800
    )
    
    return fig