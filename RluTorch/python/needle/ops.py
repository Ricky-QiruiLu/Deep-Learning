"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
from . import init
import numpy

from .backend_selection import array_api, NDArray


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple([out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(init.zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + numpy.float32(self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * numpy.float32(self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a**self.scalar
        

    def gradient(self, out_grad, node):
        return (out_grad * self.scalar * node.inputs[0]**(self.scalar - 1), )
        


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        
        return a / b
        

    def gradient(self, out_grad, node):
        
        lhs, rhs = node.inputs
        return  divide(out_grad, rhs), (-1 * out_grad * lhs) / rhs**2
        


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a / self.scalar
        

    def gradient(self, out_grad, node):
        return (out_grad / self.scalar, )
        


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        
        new_axes = [i for i in range(len(a.shape))]
        if self.axes:
            idx1, idx2 = self.axes
            new_axes[idx1], new_axes[idx2] = new_axes[idx2], new_axes[idx1]
        else:
            new_axes[-1] = len(a.shape) - 2
            new_axes[-2] = len(a.shape) - 1
        new_axes = tuple(new_axes)
        return a.permute(new_axes)
        

    def gradient(self, out_grad, node):
        return (transpose(out_grad, axes=self.axes), )
        


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.reshape(a, self.shape)
        

    def gradient(self, out_grad, node):
        return (reshape(out_grad, node.inputs[0].shape), )
        


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape).compact()

    def gradient(self, out_grad, node):
        
        input_matrix = node.inputs[0]
        input_shape = input_matrix.shape
        
        out_shape = out_grad.shape
        
        input_shape_list = list(input_shape)
        
        if len(out_shape) != len(input_shape):
            input_shape_list = [1] * (len(out_shape) - len(input_shape)) + input_shape_list
            input_shape = tuple(input_shape_list)
            
        axes = []
        idx = 0
        for i, j in zip(out_shape, input_shape):
            if i != j:
                axes.append(idx)
            idx += 1
            
        axes = tuple(axes)
        result = summation(out_grad, axes)
        return (reshape(result, input_matrix.shape), )
        


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return array_api.summation(a, axis=self.axes)
        

    def gradient(self, out_grad, node):
        input_matrix = node.inputs[0]

        if self.axes is None:
            new_shape = list(input_matrix.shape)
            for i in range(len(new_shape)):
                new_shape[i] = 1
            new_shape = tuple(new_shape)
            return (broadcast_to(out_grad.reshape(new_shape), input_matrix.shape), )

        new_shape = list(input_matrix.shape)
        
        if len(out_grad.shape) <= 0:
            return (broadcast_to(out_grad, input_matrix.shape), )
        
        if isinstance(self.axes, int):
          new_shape[self.axes] = 1
        else:
          for i in self.axes:
              new_shape[i] = 1
        
        new_shape = tuple(new_shape)
        return (broadcast_to(reshape(out_grad, new_shape), input_matrix.shape), )
        


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        return a @ b
        

    def gradient(self, out_grad, node):
        
        lhs, rhs = node.inputs
        lhs_result = matmul(out_grad, transpose(rhs))
        rhs_result = matmul(transpose(lhs), out_grad)
        
        if len(out_grad.shape) > len(lhs.shape):
            axes = [i for i in range(len(out_grad.shape) - len(lhs.shape))]
            axes = tuple(axes)
            lhs_result = summation(lhs_result, axes)
        
        if len(out_grad.shape) > len(rhs.shape):
            axes = [i for i in range(len(out_grad.shape) - len(rhs.shape))]
            axes = tuple(axes)
            rhs_result = summation(rhs_result, axes)
        
        return lhs_result, rhs_result
        


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        
        return -a
        

    def gradient(self, out_grad, node):
        
        return (-1 * out_grad, )
        


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        
        return array_api.log(a)
        

    def gradient(self, out_grad, node):
        
        return (divide(out_grad, node.inputs[0]), )
        


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        
        return array_api.exp(a)
        

    def gradient(self, out_grad, node):
        
        return (multiply(out_grad, exp(node.inputs[0])), )
        


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        
        return array_api.maximum(a, 0)
        

    def gradient(self, out_grad, node):
        
        temp_metrix = node.inputs[0].realize_cached_data() > 0
        return (out_grad * Tensor(temp_metrix),)
        


def relu(a):
    return ReLU()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

        if isinstance(self.axes, int):
          self.axes = (self.axes, )

    def compute(self, Z):
        
        max_z = array_api.amax(Z, axis=self.axes, keepdims=True)
        return array_api.log(array_api.summation(array_api.exp(Z - max_z.broadcast_to(Z.shape)), axis=self.axes)) + array_api.amax(Z, axis=self.axes)
        

    def gradient(self, out_grad, node):
        
        input_node = node.inputs[0]
        
        if not self.axes is None:
            shape = [1] * len(input_node.shape)
            idx = 0
            for i in range(len(shape)):
                if i not in self.axes:
                    shape[i] = node.shape[idx]
                    idx += 1
            
            output = node.reshape(shape).broadcast_to(input_node.shape)
            out_grad_bc = out_grad.reshape(shape).broadcast_to(input_node.shape)
            
        else:
            shape = input_node.shape
            output = node.broadcast_to(input_node.shape)
            out_grad_bc = out_grad.broadcast_to(input_node.shape)
            
        softmax = exp(input_node - output)
        return (softmax * out_grad_bc, )
        


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)


class Tanh(TensorOp):
    def compute(self, a):
        
        return array_api.tanh(a)
        

    def gradient(self, out_grad, node):
        
        input_node = node.inputs[0]
        return (negate((1 - tanh(input_node)**2) * out_grad), )
        


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
    # def compute(self, args: tuple(NDArray)) -> NDArray:
        
        
        # get the shape of the output
        n = len(args)
        new_shape = list(args[0].shape)
        new_shape.insert(self.axis, len(args))
        new_shape = tuple(new_shape)
        
        # create the output and fillin the values
        result = array_api.empty(new_shape, device=args[0].device)

        for i, temp in enumerate(args):
          slices = []
          for j in range(len(new_shape)):
            if j == self.axis:
              slices.append(slice(i, i + 1, 1))
            else:
              slices.append(slice(0, new_shape[j], 1))
          slices = tuple(slices)

          result[slices] = temp

        return result
        

    def gradient(self, out_grad, node):
        
        out_split = split(out_grad, node.op.axis)
        return (out_split, )
        


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        
        result = []
        new_shape = list(A.shape)
        del new_shape[self.axis]

        for i in range(A.shape[self.axis]):
          slices = []
          for j in range(len(A.shape)):
            if j == self.axis:
              slices.append(slice(i, i + 1, 1))
            else:
              slices.append(slice(0, A.shape[j], 1))
          slices = tuple(slices)

          temp = A[slices].compact().reshape(new_shape)
          result.append(Tensor(temp, device=A.device))
        
        result2 = tuple(result)


        return tuple(result)
        

    def gradient(self, out_grad, node):
        

        # stack the out_grad
        result = stack(out_grad, node.op.axis)
        return (result, )
        


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        
        return array_api.flip(a, self.axes)
        

    def gradient(self, out_grad, node):
        
        
        return (flip(out_grad, self.axes), )
        


def flip(a, axes):
    return Flip(axes)(a)



class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        

        # new shape
        new_shape = list(a.shape)
        for i in self.axes:
          if i > len(a.shape) - 1:
            continue

          new_shape[i] = new_shape[i] * (1 + self.dilation)
        new_shape = tuple(new_shape)

        # get the proper slices for new array
        new_array = array_api.full(new_shape, 0, device=a.device)
        
        slices = []
        for i in range(len(new_shape)):
          if i not in self.axes:
            slices.append(slice(0, new_shape[i], 1))
          else:
            slices.append(slice(0, new_shape[i], 1 + self.dilation))
        slices = tuple(slices)

        new_array.__setitem__(slices, a)

        return new_array
        

    def gradient(self, out_grad, node):
        
        input_node = node.inputs[0]
        undilate_out_grad = undilate(out_grad, self.axes, self.dilation)
        return (undilate_out_grad, )
        


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)

class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        
        
        # new shape
        new_shape = list(a.shape)
        for i in self.axes:
          if i > len(a.shape) - 1:
            continue
          new_shape[i] = new_shape[i] // (1 + self.dilation)
        new_shape = tuple(new_shape)

        # get the proper slices for new array
        slices = []
        for i in range(len(new_shape)):
          if i not in self.axes:
            slices.append(slice(0, a.shape[i], 1))
          else:
            slices.append(slice(0, a.shape[i], 1 + self.dilation))
        slices = tuple(slices)

        new_np_array = a[slices]
        return new_np_array

        

    def gradient(self, out_grad, node):
        
        dilate_out_grad = dilate(out_grad, self.axes, self.dilation)
        return (dilate_out_grad, )
        


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)

class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        # padding
        if self.padding:
          n = self.padding
          A_ps = A.pad(((0, 0),(n, n),(n, n),(0, 0)))
        
        else:
          A_ps = A

        # get the shapes
        N,H,W,C_in = A_ps.shape
        K,_,_,C_out = B.shape
        Ns, Hs, Ws, Cs = A_ps.strides
        inner_dim = K * K * C_in

        T_H = H - K + 1
        T_W = W - K + 1

        if self.stride > 1:
          T_H = math.ceil(T_H / self.stride)
          T_W = math.ceil(T_W / self.stride)

        A_ps = A_ps.as_strided(shape=(N, T_H, T_W, K, K, C_in),
                                            strides = (Ns, Hs * self.stride, Ws * self.stride, Hs, Ws, Cs)).compact()
        A_ps = A_ps.reshape((N * T_H * T_W, inner_dim))

        # print("shapes", A.shape, A_ps.shape, B.shape, )
        B = B.compact()
        out = A_ps @ B.reshape((inner_dim, C_out))

        return out.reshape((N,T_H,T_W,C_out))

    def gradient(self, out_grad, node):
        X, w = node.inputs
        k = w.shape[0]

        # X.grad
        w_flip = transpose(flip(w, axes=(0, 1)), axes=(2,3))

        if self.stride > 1:
          out_grad_dila = dilate(out_grad, (1, 2), dilation=self.stride-1)
        else:
          out_grad_dila = out_grad

        X_grad = conv(out_grad_dila, w_flip, stride=1, padding=k - 1 - self.padding)
        w_grad = conv(transpose(X, axes=(0,3)), transpose(transpose(out_grad_dila, axes=(0,1)), axes=(1,2)), stride=1, padding=self.padding)
        w_grad = transpose(transpose(w_grad, axes=(0,1)), axes=(1,2))
        return (X_grad, w_grad)


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)

