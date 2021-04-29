from __future__ import absolute_import, print_function

import tvm
import numpy as np
import tvm.topi as topi

# Global declarations of environment.

# llvm
tgt_host="llvm"
# llvm, cuda, opencl, metal
# Change it to respective GPU if gpu is enabled Ex: cuda, opencl
tgt="llvm"


def make_elemwise_add(shape, tgt, tgt_host, func_name, dtype="float32"):
    A = tvm.te.placeholder(shape, dtype=dtype, name="A")
    B = tvm.te.placeholder(shape, dtype=dtype, name="B")
    C = tvm.te.compute(A.shape, lambda *i: A(*i) + B(*i))

    s = tvm.te.create_schedule(C.op)
    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_elemwise_mul(shape, tgt, tgt_host, func_name, dtype="float32"):
    """TODO: Your code here"""
    A = tvm.placeholder(shape, dtype=dtype, name='A')
    B = tvm.placeholder(shape, dtype=dtype, name='B')
    C = tvm.compute(A.shape, lambda *i: A(*i) * B(*i))

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
    return f

def make_elemwise_add_by_const(shape, const_k, tgt, tgt_host, func_name,
                               dtype="float32"):
    """TODO: Your code here"""
    A = tvm.placeholder(shape, dtype=dtype, name='A')
    B = tvm.compute(A.shape, lambda *i: A(*i) + const_k)

    s = tvm.create_schedule(B.op)
    f = tvm.build(s, [A, B], tgt, target_host=tgt_host, name=func_name)
    return f


def make_elemwise_mul_by_const(shape, const_k, tgt, tgt_host, func_name,
                            dtype="float32"):
    """TODO: Your code here"""
    A = tvm.placeholder(shape, dtype=dtype, name='A')
    B = tvm.compute(A.shape, lambda *i: A(*i) * const_k)

    s = tvm.create_schedule(B.op)
    f = tvm.build(s, [A, B], tgt, target_host=tgt_host, name=func_name)
    return f

def make_relu(shape, tgt, tgt_host, func_name, dtype="float32"):
    """TODO: Your code here"""
    """Hint: use tvm.max, tvm.const(0, A.dtype)"""
    A = tvm.placeholder(shape, dtype=dtype, name='A')
    B = tvm.compute(A.shape, lambda *i: tvm.max(A(*i), tvm.const(0, A.dtype)))

    s = tvm.create_schedule(B.op)
    f = tvm.build(s, [A, B], tgt, target_host=tgt_host, name=func_name)
    return f


def make_relu_gradient(shape, tgt, tgt_host, func_name, dtype="float32"):
    """TODO: Your code here"""
    """Hint: use tvm.select"""
    input_val = tvm.placeholder(shape, dtype=dtype, name='input_val')
    output_grad = tvm.placeholder(shape, dtype=dtype, name='output_grad')
    A = tvm.tir.Select(input_val > 0, 1, 0)
    B = tvm.compute(A.shape, lambda *i: A(*i) * output_grad(*i))

    s = tvm.create_schedule(B.op)
    f = tvm.build(s, [input_val, output_grad], tgt, target_host=tgt_host, name=func_name)
    return f


def make_matrix_mul(shapeA, transposeA, shapeB, transposeB, tgt, tgt_host,
                    func_name, dtype="float32"):
    """TODO: Your code here"""
    """Hint: use tvm.reduce_axis, tvm.sum"""
    """Hint: treat 4 cases of transposeA, transposeB separately"""
    """Hint: for tvm schedule, use split, reorder, vectorize, parallel"""
    """Hint: debug tvm schedule using tvm.lower"""
    A = tvm.placeholder(shapeA, dtype=dtype, name='A')
    B = tvm.placeholder(shapeB, dtype=dtype, name='B')
    if not transposeA and not transposeB:
        # A x B
        target_shape = (shapeA[0], shapeB[1])
        k = tvm.reduce_axis((0, shapeA[1]), name='k')
        mat_mul = lambda i, j: tvm.sum(A[i, k] * B[k, j], axis=k)
        C = tvm.compute(target_shape, mat_mul)
    elif not transposeA and transposeB:
        # A x B.T
        target_shape = (shapeA[0], shapeB[0])
        k = tvm.reduce_axis((0, shapeA[1]), name='k')
        mat_mul = lambda i, j: tvm.sum(A[i, k] * B[j, k], axis=k)
        C = tvm.compute(target_shape, mat_mul)
    elif transposeA and not transposeB:
        # A.T x B
        target_shape = (shapeA[1], shapeB[1])
        k = tvm.reduce_axis((0, shapeA[0]), name='k')
        mat_mul = lambda i, j: tvm.sum(A[k, i] * B[k, j], axis=k)
        C = tvm.compute(target_shape, mat_mul)
    else:
        # A.T x B.T
        target_shape = (shapeA[1], shapeB[0])
        k = tvm.reduce_axis((0, shapeA[0]), name='k')
        mat_mul = lambda i, j: tvm.sum(A[k, i] * B[j, k], axis=k)
        C = tvm.compute(target_shape, mat_mul)
    
    s = tvm.create_schedule(C.op) # Default scheduler
    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
    return f



def make_conv2d(shapeX, shapeF, tgt, tgt_host, func_name, dtype="float32"):
    assert(shapeX[1] == shapeF[1])
    N, C, H, W = shapeX
    M, C, R, S = shapeF

    """TODO: Your code here"""
    """Hint: use tvm.reduce_axis, tvm.sum"""
    """Hint: go by conv2d definition. Treat stride=1, padding=0 case only."""
    """For a challenge, treat the general case for stride and padding."""
    X = tvm.placeholder(shapeX, dtype=dtype, name='X')
    F = tvm.placeholder(shapeF, dtype=dtype, name='F')
    target_shape = (N, M, H - R + 1, W - S + 1)
    kc = tvm.reduce_axis((0, C), name='kc')
    kr = tvm.reduce_axis((0, R), name='kr')
    ks = tvm.reduce_axis((0, S), name='ks')
    C = tvm.compute(target_shape, lambda i, j, k, l: tvm.sum(X[i, kc, k + kr, l + ks] * F[j, kc, kr, ks], axis=(kc, kr, ks)))

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [X, F, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_matrix_softmax(shape, tgt, tgt_host, func_name, dtype="float32"):

    """TODO: Your code here"""
    """Hint: use tvm.reduce_axis, tvm.sum, tvm.max, tvm.exp"""
    """Hint: do not reuse the same reduction axis j."""
    """Hint: implement the following version for better stability
        e_x = np.exp(x - np.max(x))
        softmax(x)= e_x / e_x.sum()
    """

    assert len(shape) == 2
    A = tvm.placeholder(shape, dtype=dtype, name='A')
    j_max = tvm.reduce_axis((0, shape[1]), name='j_max')
    j_sum = tvm.reduce_axis((0, shape[1]), name='j_sum')
    A_exp = tvm.compute(shape, lambda i, j: tvm.exp(A[i, j] - tvm.max(A[i, j_max], axis=j_max)))
    B = tvm.compute(shape, lambda i, j: A_exp[i, j] / tvm.sum(A_exp[i, j_sum], axis=j_sum))

    s = tvm.create_schedule(B.op)
    f = tvm.build(s, [A], tgt, target_host=tgt_host, name=func_name)
    return f


def make_matrix_softmax_cross_entropy(shape, tgt, tgt_host, func_name,
                                      dtype="float32"):
    """TODO: Your code here"""
    """Hint: output shape should be (1,)"""
    
    assert len(shape) == 2
    A = tvm.placeholder(shape, dtype=dtype, name='A')
    B = tvm.placeholder(shape, dtype=dtype, name='B')
    j_max = tvm.reduce_axis((0, shape[1]), name='j_max')
    j_sum = tvm.reduce_axis((0, shape[1]), name='j_sum')
    A_exp = tvm.compute(shape, lambda i, j: tvm.exp(A[i, j] - tvm.max(A[i, j_max], axis=j_max)))
    C = tvm.compute(shape, lambda i, j: A_exp[i, j] / tvm.sum(A_exp[i, j_sum], axis=j_sum))
    
    i_sum_loss = tvm.reduce_axis((0, shape[0]), name='i_sum_loss')
    j_sum_loss = tvm.reduce_axis((0, shape[1]), name='j_sum_loss')
    D = tvm.compute((1,), lambda *i: -tvm.sum(tvm.log(A[i_sum_loss, j_sum_loss]) * B[i_sum_loss, j_sum_loss], axis=(i_sum_loss, j_sum_loss)))

    s = tvm.create_schedule(D.op)
    f = tvm.build(s, [A, B], tgt, target_host=tgt_host, name=func_name)
    return f


def make_reduce_sum_axis_zero(shape, tgt, tgt_host, func_name, dtype="float32"):
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    C = topi.sum(A, axis=0, keepdims=False)

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_broadcast_to(shape, to_shape, tgt, tgt_host, func_name,
                      dtype="float32"):
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    C = topi.broadcast_to(A, to_shape)

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_sgd_update(shape, learning_rate, tgt, tgt_host, func_name,
                    dtype="float32"):
    X = tvm.placeholder(shape, dtype=dtype, name="A")
    grad = tvm.placeholder(shape, dtype=dtype, name="grad")
    Y = tvm.compute(shape, lambda *i: X(*i) - learning_rate * grad(*i))

    s = tvm.create_schedule(Y.op)
    f = tvm.build(s, [X, grad, Y], tgt, target_host=tgt_host, name=func_name)
    return f