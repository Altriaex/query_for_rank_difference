# coding: utf-8
import tensorflow.compat.v1 as tf
from .sigmoidal_utility_I import  _shift_item_scores, _log_sigmoidal_utility
from .sigmoidal_utility_I import create_train_loss_op as __create_train_loss_op_I

def create_train_loss_op(place_holders, params, l2_reg, lr_t, step_t):
    op1_I, loss1_I, op2_I, loss2_I = __create_train_loss_op_I(place_holders, params,
                                                              l2_reg, lr_t, step_t)
    op1_II, loss1_II = _create_train_op1_II(place_holders, params, l2_reg, lr_t, _log_sigmoidal_utility)
    op2_II, loss2_II = _create_train_op2_II(place_holders, params, l2_reg, lr_t, _log_sigmoidal_utility)
    loss1 = loss1_I + loss1_II
    loss2 = loss2_I + loss2_II
    op1 = tf.group(op1_I, op1_II)
    op2 = tf.group(op2_I, op2_II)
    return op1, loss1, op2, loss2

def sigmoid_utility_tf(x, c1, c2):
    return 1. / (1 + tf.exp(tf.reduce_sum(-c1*(x-c2), axis=1)))


def _compute_typeII_loss(place_holders, params, log_util_fn):
    label1 = place_holders["label1"]
    label2 = place_holders["label2"]
    label_type2 = place_holders["label_type2"]
    c1 = tf.gather(params["c1"], place_holders["worker_II"])
    c2 = tf.gather(params["c2"], place_holders["worker_II"])
    item11_emb = tf.gather(params["item"], place_holders["item11"])
    item11_util = tf.exp(log_util_fn(item11_emb, place_holders, params))
    item12_emb = tf.gather(params["item"], place_holders["item12"])
    item12_util = tf.exp(log_util_fn(item12_emb, place_holders, params))
    util_diff1 = label1 * (item11_util - item12_util) \
                   + (1. - label1) * (item12_util - item11_util)
    item21_emb = tf.gather(params["item"], place_holders["item21"])
    item21_util = tf.exp(log_util_fn(item21_emb, place_holders, params))
    item22_emb = tf.gather(params["item"], place_holders["item22"])
    item22_util = tf.exp(log_util_fn(item22_emb, place_holders, params))
    util_diff2 = label2 * (item21_util - item22_util) \
                   + (1. - label2) * (item22_util - item21_util)
    loss = (1 - label_type2) * tf.nn.relu(util_diff1 - util_diff2)\
                + label_type2 * tf.nn.relu(util_diff2 - util_diff1)
    loss = tf.reduce_mean(loss)
    return loss

def _create_train_op1_II(place_holders, params, l2_reg, lr_t, log_util_fn):
    loss = _compute_typeII_loss(place_holders, params, log_util_fn)
    train_step = tf.train.AdamOptimizer(lr_t).minimize(loss,
                    var_list=tf.trainable_variables("worker_param/item"))
    return train_step, loss

def _create_train_op2_II(place_holders, params, l2_reg, lr_t, log_util_fn):
    loss = _compute_typeII_loss(place_holders, params, log_util_fn)
    train_step = tf.train.AdamOptimizer(lr_t).minimize(loss,
                    var_list=tf.trainable_variables("worker_param/c1")\
                        + tf.trainable_variables("worker_param/c2"))
    assign_op = _shift_item_scores(params)
    return tf.group(train_step, assign_op), loss