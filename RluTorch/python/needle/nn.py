"""The module.
"""
from typing import List
from typing_extensions import Required
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype))

        if bias:
            self.bias = Parameter(ops.transpose(init.kaiming_uniform(out_features, 1, dtype=dtype, device=device, requires_grad=True)))
        else:
            self.bias = None
        
    def forward(self, X: Tensor) -> Tensor:
        if self.bias is None:
          return  X @ self.weight
        
        # has bias
        if len(X.shape) > 2:
          m = 1
          for i in X.shape[1:]:
            m *= i
          X_reshape = X.reshape((X.shape[0], m))
        else:
          X_reshape = X
        
        result = X_reshape @ self.weight
        new_shape = [1 for _ in result.shape]
        new_shape[-1] = self.out_features
        new_shape = tuple(new_shape)
        result += self.bias.reshape(new_shape).broadcast_to(result.shape)      # broadcast the bias
            
        return result
        

class Flatten(Module):
    def forward(self, X):
        X_shape = X.shape
        if len(X_shape) == 2:
            return X
            
        m = 1
        for i in range(len(X_shape) - 1):
            m *= X_shape[i + 1]
        
        return X.reshape((X_shape[0], m))
        

class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)
        

class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.tanh(x)
        

class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return init.ones(*(x.shape), device=x.device, dtype=x.dtype, requires_grad=True) / (1 + ops.exp(-x))
        

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        result = x
        for curr_module in self.modules:
            result = curr_module(result)
        return result
        

class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        Zy = init.one_hot(logits.shape[1], y, dtype=logits.dtype, device=logits.device)
        log_sum = ops.log(ops.summation(ops.exp(logits), axes=(1,))) - ops.summation(Zy * logits, axes=(1,))
        return log_sum.sum() / log_sum.shape[0]
        

class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        
    def forward(self, x: Tensor) -> Tensor:
        shape = (1, self.dim)
        if self.training:
            # batch normalization
            # Exp(x)
            E_x = x.sum(axes=(0,)) / x.shape[0]
            E_x = E_x.reshape(shape).broadcast_to(x.shape)
            
            # Var(x)
            Var_x = ((x - E_x)**2).sum(axes=(0,))
            Var_x = Var_x / x.shape[0]
            
            deno = (Var_x + self.eps)**0.5
            x_normalized = (x - E_x) / deno.reshape(shape).broadcast_to(x.shape)
            
            
            # update the running estimates
            curr_running_mean = (x.sum(axes=(0,)) / x.shape[0]).data

            curr_running_var = (((x - curr_running_mean.reshape(shape).broadcast_to(x.shape))**2).sum(axes=(0,)) / x.shape[0]).data
            
            self.running_mean  = (1 - self.momentum) * self.running_mean + self.momentum * curr_running_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * curr_running_var
        
        else:
            deno = (self.running_var + self.eps)**0.5
            x_normalized = (x - self.running_mean) / deno.reshape(shape).broadcast_to(x.shape)
            
        return self.weight.reshape(shape).broadcast_to(x.shape) * x_normalized + self.bias.reshape(shape).broadcast_to(x.shape)

        

class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        
    def forward(self, x: Tensor) -> Tensor:
        shape = (x.shape[0], 1)
        
        # Exp(x)
        E_x = x.sum(axes=(1,)) / x.shape[1]
        # if self.dim > 1:
        E_x = E_x.reshape(shape).broadcast_to(x.shape)
            
        
        # Var(x)
        Var_x = ((x - E_x)**2).sum(axes=(1,))
        Var_x = Var_x / x.shape[1]
        
        # normalized x
        denominator = ((Var_x + self.eps)**0.5)
        denominator = denominator.reshape(shape).broadcast_to(x.shape)
        
        x_normalized = (x - E_x) / denominator
        
        result = self.weight.reshape((1, self.dim)).broadcast_to(x.shape) * x_normalized + self.bias.reshape((1, self.dim)).broadcast_to(x.shape)
        
        return result
        

class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            rand_matrix = init.randb(*x.shape, p=(1 - self.p))
            return x * rand_matrix / (1 - self.p)
        
        return x
        

class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(x) + x
        
class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        self.device = device
        self.dtype = dtype


        fan_in = self.in_channels * self.kernel_size**2
        fan_out = self.out_channels * self.kernel_size**2
        shape = (self.kernel_size, self.kernel_size, self.in_channels, self.out_channels)

        self.weight = Parameter(init.kaiming_uniform(fan_in, fan_out, shape=shape, device=device, dtype=dtype, requires_grad=True))
        bound = 1.0 / (self.in_channels * self.kernel_size**2)**0.5
        
        self.bias = None
        if bias:
          self.bias = Parameter(init.rand(self.out_channels, low=-bound, high=bound, device=device, dtype=dtype))
        
    def forward(self, x: Tensor) -> Tensor:
        # storage order: batch height width channels / NHWC, but in pytorch NCHW
        # weight: kernelSize kernelSize in_channel out_channel, pytorch is: in out ker ker

        # shape of x
        x = x.transpose(axes=(1,2)).transpose(axes=(2,3))

        # padding
        num_padding = (self.kernel_size-1) // 2

        result = ops.conv(x, self.weight, stride=self.stride, padding=num_padding)
        
        if not self.bias is None:
          result += self.bias.reshape((1, 1, 1, self.out_channels)).broadcast_to(result.shape)
        
        return result.transpose((1, 3)).transpose((2, 3))
        

class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.
        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.
        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).
        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity
        self.device = device
        self.dtype = dtype

        k = 1 / hidden_size
        self.W_ih = Parameter(init.rand(input_size, hidden_size, low=-k**0.5, high=k**0.5, device=device, dtype=dtype, requires_grad=True))
        self.W_hh = Parameter(init.rand(hidden_size, hidden_size, low=-k**0.5, high=k**0.5, device=device, dtype=dtype, requires_grad=True))

        self.bias_ih = None
        self.bias_hh = None
        if self.bias:
          self.bias_ih = Parameter(init.rand(hidden_size, low=-k**0.5, high=k**0.5, device=device, dtype=dtype, requires_grad=True)) 
          self.bias_hh = Parameter(init.rand(hidden_size, low=-k**0.5, high=k**0.5, device=device, dtype=dtype, requires_grad=True))
        
        
    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.
        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        if h is None:
            h = init.zeros(X.shape[0], self.hidden_size, device=self.device, dtype=self.dtype)

        xW_ih = X @ self.W_ih
        hW_hh = h @ self.W_hh

        if not self.bias:
          if self.nonlinearity == 'tanh':
            return ops.tanh(xW_ih + hW_hh)

          if self.nonlinearity == 'relu':
            return ops.relu(xW_ih + hW_hh)

        bias_ih_bc = self.bias_ih.reshape((1, self.hidden_size)).broadcast_to((X.shape[0], self.hidden_size))
        bias_hh_bc = self.bias_hh.reshape((1, self.hidden_size)).broadcast_to((X.shape[0], self.hidden_size))

        if self.nonlinearity == 'tanh':
          return ops.tanh(xW_ih + hW_hh + bias_ih_bc + bias_hh_bc)
        elif self.nonlinearity == 'relu':
          return ops.relu(xW_ih + hW_hh + bias_ih_bc + bias_hh_bc)
    
        

class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.
        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.
        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.nonlinearity = nonlinearity
        self.rnn_cells = []
        for i in range(self.num_layers):
            if i == 0:
                self.rnn_cells.append(RNNCell(input_size, hidden_size, bias, nonlinearity, device, dtype))
            else:
                self.rnn_cells.append(RNNCell(hidden_size, hidden_size, bias, nonlinearity, device, dtype))
        
    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.
        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        if h0 is None:
            h0 = init.zeros(self.num_layers, X.shape[1], self.hidden_size, device=X.device, dtype=X.dtype)

        result = []
        X_split = ops.split(X, axis=0)
        for i in range(X.shape[0]):
            h0_split = list(ops.split(h0, axis=0))

            for j in range(self.num_layers):
                if j == 0:
                    h0_split[j] = self.rnn_cells[j](X_split[i], h0_split[j])
                else:
                    h0_split[j] = self.rnn_cells[j](h0_split[j - 1], h0_split[j])
            
            result.append(h0_split[-1])
            h0 = ops.stack(h0_split, axis=0)

        return ops.stack(result, axis=0), h0
        

class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.device = device
        self.dtype = dtype

        k = 1 / hidden_size
        self.W_ih = Parameter(init.rand(input_size, 4 * hidden_size, low=-k** 0.5, high=k** 0.5, device=device, dtype=dtype, requires_grad=True))
        self.W_hh = Parameter(init.rand(hidden_size, 4 * hidden_size, low=-k**0.5, high=k** 0.5, device=device, dtype=dtype, requires_grad=True))

        self.bias_ih = None
        self.bias_hh = None
        if self.bias:
          self.bias_ih = Parameter(init.rand(4 * hidden_size, low=-k**0.5, high=k**0.5, device=device, dtype=dtype, requires_grad=True)) 
          self.bias_hh = Parameter(init.rand(4 * hidden_size, low=-k**0.5, high=k**0.5, device=device, dtype=dtype, requires_grad=True))
        

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        if h is None:
            h = (init.zeros(X.shape[0], self.hidden_size, device=X.device, dtype=X.dtype), init.zeros(X.shape[0], self.hidden_size, device=X.device, dtype=X.dtype))
        
        h0, c0 = h

        temp = X @ self.W_ih + h0 @ self.W_hh
        if self.bias:
            bias_ih_bc = self.bias_ih.reshape((1, self.bias_ih.shape[0])).broadcast_to(temp.shape)
            bias_hh_bc = self.bias_hh.reshape((1, self.bias_hh.shape[0])).broadcast_to(temp.shape)

            temp += bias_ih_bc + bias_hh_bc

        temp_split = tuple(ops.split(temp, axis=1))
      
        gates = []
        for i in range(4):
          curr_gate = ops.stack(temp_split[self.hidden_size*i:self.hidden_size*(i+1)], axis=1)

          if i == 2:
            gates.append(Tanh()(curr_gate))
          else:
            gates.append(Sigmoid()(curr_gate))

        i, f, g, o = tuple(gates)
        c_p = f * c0 + i * g
        h_p = o * Tanh()(c_p)

        return h_p, c_p
        

class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.device = device
        self.dtype = dtype

        self.lstm_cells = []
        for i in range(self.num_layers):
            if i == 0:
                self.lstm_cells.append(LSTMCell(input_size, hidden_size, bias, device, dtype))
            else:
                self.lstm_cells.append(LSTMCell(hidden_size, hidden_size, bias, device, dtype))
        
    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        if h is None:
          h = (init.zeros(self.num_layers, X.shape[1], self.hidden_size, device=X.device, dtype=X.dtype), 
                init.zeros(self.num_layers, X.shape[1], self.hidden_size, device=X.device, dtype=X.dtype))

        h_layers = list(ops.split(h[0], axis=0))
        c_layers = list(ops.split(h[1], axis=0))

        output = []
        X_split = ops.split(X, axis=0)

        for i in range(X.shape[0]):
          h_layers[0], c_layers[0] = self.lstm_cells[0](X_split[i], (h_layers[0], c_layers[0]))

          for j in range(1, self.num_layers):
            h_layers[j], c_layers[j] = self.lstm_cells[j](h_layers[j-1], (h_layers[j], c_layers[j]))

          output.append(h_layers[-1])

        output = ops.stack(output, axis=0)
        h_n = ops.stack(h_layers, axis=0)
        c_n = ops.stack(c_layers, axis=0)
        
        return output, (h_n, c_n)
        
