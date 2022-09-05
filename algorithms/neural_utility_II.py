# coding: utf-8
import tensorflow.compat.v1 as tf
from .neural_utility_I import _log_neural_utility
from .neural_utility_I import create_train_loss_op as __create_train_loss_op_I
from .sigmoidal_utility_II import _create_train_op1_II, _create_train_op2_II

def create_train_loss_op(place_holders, params, l2_reg, lr_t, step_t):
    op1_I, loss1_I, op2_I, loss2_I = __create_train_loss_op_I(place_holders, params,
                                                              l2_reg, lr_t, step_t)
    op1_II, loss1_II = _create_train_op1_II(place_holders, params, l2_reg, lr_t, _log_neural_utility)
    op2_II, loss2_II = _create_train_op2_II(place_holders, params, l2_reg, lr_t, _log_neural_utility)
    loss1 = loss1_I + loss1_II
    loss2 = loss2_I + loss2_II
    op1 = tf.group(op1_I, op1_II)
    op2 = tf.group(op2_I, op2_II)
    return op1, loss1, op2, loss2
