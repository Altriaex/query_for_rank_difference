# coding: utf-8
import tensorflow.compat.v1 as tf
from .sigmoidal_utility_I import _shift_item_scores

def create_train_loss_op(place_holders, params, l2_reg, lr_t, step_t):
    op1, loss1 = __create_train_op1(place_holders, params, lr_t, step_t)
    op2, loss2 = __create_train_op2(place_holders, params, lr_t)
    return op1, loss1, op2, loss2

def __compute_loss(place_holders, params):
    label = place_holders["label"]
    gamma = tf.gather(params["c1"], place_holders["worker"])
    item1_score = tf.gather(params["item"], place_holders["item1"])
    item2_score = tf.gather(params["item"], place_holders["item2"])
    score_diff = tf.squeeze(gamma*(item1_score-item2_score), axis=1)
    log_prob12 = tf.log_sigmoid(score_diff)
    log_prob21 = tf.log_sigmoid(-score_diff)
    loss = -(label * log_prob12 + (1.0 - label) * log_prob21)
    return tf.reduce_mean(loss)

def __create_train_op1(place_holders, params, lr_t, step_t):
    loss = __compute_loss(place_holders, params)
    train_step = tf.train.AdamOptimizer(lr_t).minimize(
                loss, var_list=tf.trainable_variables("worker_param/item"), global_step=step_t)
    return train_step, loss

def __create_train_op2(place_holders, params, lr_t):
    loss = __compute_loss(place_holders, params)
    loss = tf.reduce_mean(loss)
    train_step = tf.train.AdamOptimizer(lr_t).minimize(
                loss, var_list=tf.trainable_variables("worker_param/c1"))
    assign_op = _shift_item_scores(params)
    return tf.group(assign_op, train_step), loss