# coding: utf-8
import tensorflow.compat.v1 as tf

def create_train_loss_op(place_holders, params, l2_reg, lr_t, step_t):
    op1, loss1 = __create_train_op1(place_holders, params, lr_t, step_t)
    op2, loss2 = __create_train_op2(place_holders, params, lr_t)
    return op1, loss1, op2, loss2

def __create_train_op1(place_holders, params, lr_t, step_t):
    label = place_holders["label"]
    # compute loss
    # batchsize, embedding_dim
    item1_score = tf.gather(params["item"], place_holders["item1"])
    item2_score = tf.gather(params["item"], place_holders["item2"])
    score_diff = tf.squeeze(item1_score-item2_score, axis=1)
    log_prob12 = tf.log_sigmoid(score_diff)
    log_prob21 = tf.log_sigmoid(-score_diff)
    loss = -(label * log_prob12 + (1.0 - label) * log_prob21)
    loss = tf.reduce_mean(loss)
    train_step = tf.train.AdamOptimizer(lr_t).minimize(
                loss, var_list=tf.trainable_variables(), global_step=step_t)
    return train_step, loss

def __create_train_op2(place_holders, params, lr_t):
    label1 = place_holders["label1"]
    label2 = place_holders["label2"]
    label_type2 = place_holders["label_type2"]
    item11_score = tf.exp(tf.gather(params["item"], place_holders["item11"]))
    item12_score = tf.exp(tf.gather(params["item"], place_holders["item12"]))
    score_diff1 = tf.squeeze(item11_score-item12_score, axis=1)
    score_diff1 = label1 * score_diff1 + (1. - label1) * (-score_diff1)
    item21_score= tf.exp(tf.gather(params["item"], place_holders["item21"]))
    item22_score = tf.exp(tf.gather(params["item"], place_holders["item22"]))
    score_diff2 = tf.squeeze(item21_score-item22_score, axis=1)
    score_diff2 = label2 * score_diff2 + (1. - label2) * (-score_diff2)
    loss = (1 - label_type2) * tf.nn.relu(score_diff1 - score_diff2)\
                + label_type2 * tf.nn.relu(score_diff2 - score_diff1)
    loss = tf.reduce_mean(loss)
    train_step = tf.train.AdamOptimizer(lr_t).minimize(loss,
                    var_list=tf.trainable_variables())
    return train_step, loss