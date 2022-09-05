# coding: utf-8
import tensorflow.compat.v1 as tf
from .sigmoidal_utility_I import _create_train_op1_I, _create_train_op2_I

def create_train_loss_op(place_holders, params, l2_reg, lr_t, step_t):
    op1, loss1 = _create_train_op1_I(place_holders, params, l2_reg, lr_t, step_t, _log_neural_utility)
    op2, loss2 = _create_train_op2_I(place_holders, params, l2_reg, lr_t, _log_neural_utility)
    return op1, loss1, op2, loss2

def _log_neural_utility(item, place_holders, params):
    c1 = tf.gather(params["c1"], place_holders["worker"])
    # batch, dim, k   batch, k
    w1, b1 = c1[:, :-1], c1[:, -1]
    c2 = tf.gather(params["c2"], place_holders["worker"])
    # batch, k  batch, 
    w2, b2 = c2[:, :-1], c2[:, -1]
    x = item[:, :, None]
    x = tf.sigmoid(tf.reduce_sum(x * w1, axis=1) + b1)
    x = tf.log_sigmoid(tf.reduce_sum(x * w2, axis=1) + b2)
    return x