# coding: utf-8
import tensorflow.compat.v1 as tf
from .crowd_bt_I import create_train_loss_op as __create_train_loss_op_I
from .bt import create_train_loss_op as __create_train_loss_op_II
def create_train_loss_op(place_holders, params, l2_reg, lr_t, step_t):
    op1, loss1, op2_I, loss2_I = __create_train_loss_op_I(place_holders, params,
                                                              l2_reg, lr_t, step_t)
    _, _, op2_II, loss2_II = __create_train_loss_op_II(place_holders, params,
                                                              l2_reg, lr_t, step_t)
    loss2 = loss2_I + loss2_II
    op2 = tf.group(op2_I, op2_II)          
    return op1, loss1, op2, loss2

