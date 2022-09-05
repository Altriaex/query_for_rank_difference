# coding: utf-8
import tensorflow.compat.v1 as tf

def create_train_loss_op(place_holders, params, l2_reg, lr_t, step_t):
    op1, loss1 = __create_train_op1(place_holders, params, l2_reg, lr_t, step_t)
    op2, loss2 = __create_train_op2(place_holders, params, lr_t)
    return op1, loss1, op2, loss2

def _compute_typeI_loss(place_holders, params):
    label = place_holders["label"]
    # 复用c1
    reliability_logit = tf.gather(params["c1"], place_holders["worker"])
    log_reliability, log_reliability_ = tf.log_sigmoid(reliability_logit), tf.log_sigmoid(-reliability_logit)
    item1_score = tf.gather(params["item"], place_holders["item1"])
    item2_score = tf.gather(params["item"], place_holders["item2"])
    score_diff = item1_score-item2_score
    log_p_bt, log_p_bt_ = tf.log_sigmoid(score_diff), tf.log_sigmoid(-score_diff)
    
    log_sum = tf.concat([log_p_bt+log_reliability, log_p_bt_+log_reliability_], axis=1)
    log_prob12 = tf.reduce_logsumexp(log_sum, axis=1)

    log_sum = tf.concat([log_p_bt+log_reliability_, log_p_bt_+log_reliability], axis=1)
    log_prob21 = tf.reduce_logsumexp(log_sum, axis=1)
    loss = -(label * log_prob12 + (1.0 - label) * log_prob21) 
    return tf.reduce_mean(loss)

def _compute_crowd_bt_regularization(place_holders, params):
    item1_score = tf.gather(params["item"], place_holders["item1"])
    item2_score = tf.gather(params["item"], place_holders["item2"])
    reg_item1_part = tf.log_sigmoid(tf.reduce_sum(item1_score, axis=1)) + tf.log_sigmoid(tf.reduce_sum(-item1_score, axis=1))
    reg_item2_part = tf.log_sigmoid(tf.reduce_sum(item2_score, axis=1))+ tf.log_sigmoid(tf.reduce_sum(-item2_score, axis=1))
    return -tf.reduce_mean(reg_item1_part + reg_item2_part)

def __create_train_op1(place_holders, params, l2_reg, lr_t, step_t):
    loss = _compute_typeI_loss(place_holders, params)
    loss += l2_reg * _compute_crowd_bt_regularization(place_holders, params)
    train_step = tf.train.AdamOptimizer(lr_t).minimize(
                loss, var_list=tf.trainable_variables("worker_param/item"), global_step=step_t)
    return train_step, loss

def __create_train_op2(place_holders, params, lr_t):
    loss = _compute_typeI_loss(place_holders, params)
    loss = tf.reduce_mean(loss)
    train_step = tf.train.AdamOptimizer(lr_t).minimize(
                loss, var_list=tf.trainable_variables("worker_param/c1"))
    return train_step, loss
