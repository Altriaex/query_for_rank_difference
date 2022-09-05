# coding: utf-8
import tensorflow.compat.v1 as tf
from .sigmoidal_utility_I import _create_train_op1_I
from .moe_utility_I import _create_train_op2_I

def create_train_loss_op(place_holders, params, l2_reg, lr_t, step_t):
    op1, loss1 = _create_train_op1_I(place_holders, params, l2_reg, lr_t, step_t, _log_mog_utility_tf)
    op2, loss2 = _create_train_op2_I(place_holders, params, l2_reg, lr_t, _log_mog_utility_tf)
    return op1, loss1, op2, loss2

def _log_mog_utility_tf(item, place_holders, params):
    # batch, 4
    weight_logits = tf.gather(params["c1"], place_holders["worker"])
    # 4, 
    mu, sigma = params["c2"][:, 0], tf.nn.softplus(params["c2"][:, 1])
    # batch, 1 - None, 4 -> batch, 4
    log_components = -(item - mu[None, :])**2 / (2. * sigma[None, :]**2)
    log_weights = tf.nn.log_softmax(weight_logits, axis=1)
    # batch, 
    log_util = tf.reduce_logsumexp(log_components + log_weights, axis=1)
    return log_util