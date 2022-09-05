# coding: utf-8
import tensorflow.compat.v1 as tf
from .sigmoidal_utility_I import _create_train_op1_I
#_create_train_op2_I

from .sigmoidal_utility_I import _compute_typeI_loss, _shift_item_scores

def create_train_loss_op(place_holders, params, l2_reg, lr_t, step_t):
    op1, loss1 = _create_train_op1_I(place_holders, params, l2_reg, lr_t, step_t, _log_moe_utility_tf)
    op2, loss2 = _create_train_op2_I(place_holders, params, l2_reg, lr_t, _log_moe_utility_tf)
    return op1, loss1, op2, loss2

def _log_moe_utility_tf(item, place_holders, params):
    # batch, 4
    weight_logits = tf.gather(params["c1"], place_holders["worker"])
    # 4, 
    mu, sigma = params["c2"][:, 0], params["c2"][:, 1]
    # batch, 1 - None, 4 -> batch, 4
    log_components = (item - mu[None, :]) * sigma[None, :]
    log_weights = tf.nn.log_softmax(weight_logits, axis=1)
    # batch, 
    log_util = tf.reduce_logsumexp(log_components + log_weights, axis=1)
    return log_util

def _create_train_op2_I(place_holders, params, l2_reg, lr_t, log_util_fn):
    loss = _compute_typeI_loss(place_holders, params, log_util_fn)
    reg = l2_reg * tf.reduce_sum(params["c2"]**2)
    loss += reg
    train_step = tf.train.AdamOptimizer(lr_t).minimize(loss,
                    var_list=tf.trainable_variables("worker_param/c1")\
                        + tf.trainable_variables("worker_param/c2"))
    assign_op = _shift_item_scores(params)
    return tf.group(train_step, assign_op), loss
