# coding: utf-8
import tensorflow.compat.v1 as tf
from .crowd_bt_I import _compute_crowd_bt_regularization

def create_train_loss_op(place_holders, params, l2_reg, lr_t, step_t):
    op1, loss1 = _create_train_op1_I(place_holders, params, l2_reg, lr_t, step_t, _log_sigmoidal_utility)
    op2, loss2 = _create_train_op2_I(place_holders, params, l2_reg, lr_t, _log_sigmoidal_utility)
    return op1, loss1, op2, loss2

def _log_sigmoidal_utility(item, place_holders, params):
    c1 = tf.gather(params["c1"], place_holders["worker"])
    c2 = tf.gather(params["c2"], place_holders["worker"])
    logutil = tf.log_sigmoid(tf.reduce_sum((item - c2) * c1, axis=1))
    return logutil

def _compute_typeI_loss(place_holders, params, log_util_fn):
    label = place_holders["label"]
    # compute loss
    # batchsize, embedding_dim
    item1_emb = tf.gather(params["item"], place_holders["item1"])
    item2_emb = tf.gather(params["item"], place_holders["item2"])
    item1_logutil = log_util_fn(item1_emb, place_holders, params)
    item2_logutil = log_util_fn(item2_emb, place_holders, params)
    log_sum = tf.reduce_logsumexp(
        tf.concat([item1_logutil[:, None], item2_logutil[:, None]], axis=1), axis=1)
    log_prob12 = item1_logutil - log_sum
    log_prob21 = item2_logutil - log_sum
    loss = -(label * log_prob12 + (1.0 - label) * log_prob21)
    loss = tf.reduce_mean(loss)
    return loss

def _shift_item_scores(params):
    # shift item scores
    n_item = tf.shape(params["item"])[0]
    # (n_item, n_item) * (n_item, 1)
    factor = tf.eye(n_item) - tf.ones((n_item, n_item))/ tf.to_float(n_item)
    assign_op = tf.assign(params["item"], tf.matmul(factor, params["item"]))
    return assign_op

def _create_train_op1_I(place_holders, params, l2_reg, lr_t, step_t, log_util_fn):
    loss = _compute_typeI_loss(place_holders, params, log_util_fn)
    train_step = tf.train.AdamOptimizer(lr_t).minimize(
                loss, var_list=tf.trainable_variables("worker_param/item"), global_step=step_t)
    return train_step, loss

def _create_train_op2_I(place_holders, params, l2_reg, lr_t, log_util_fn):
    loss = _compute_typeI_loss(place_holders, params, log_util_fn)
    train_step = tf.train.AdamOptimizer(lr_t).minimize(loss,
                    var_list=tf.trainable_variables("worker_param/c1")\
                        + tf.trainable_variables("worker_param/c2"))
    assign_op = _shift_item_scores(params)
    return tf.group(train_step, assign_op), loss