# coding: utf-8
import tensorflow.compat.v1 as tf
from .sigmoidal_utility_I import _create_train_op1_I, _create_train_op2_I

def create_train_loss_op(place_holders, params, l2_reg, lr_t, step_t):
    op1, loss1 = _create_train_op1_I(place_holders, params, l2_reg, lr_t, step_t, _log_hbtl_utility_tf)
    op2, loss2 = _create_train_op2_I(place_holders, params, l2_reg, lr_t, _log_hbtl_utility_tf)
    return op1, loss1, op2, loss2

def _log_hbtl_utility_tf(item, place_holders, params):
    gamma = tf.gather(params["c1"], place_holders["worker"])
    logutil = tf.squeeze(gamma * item, axis=1)
    return logutil
