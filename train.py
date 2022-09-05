import os.path as osp
import os
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from set_flags import FLAGS
from absl import app
import data_utils
import train_utils
import algorithms
import numpy as np

def create_model_dir(exp_name, method):
    model_dir = ""
    for prefix in ["experiments", exp_name, method]:
        model_dir = osp.join(model_dir, prefix)
        if not osp.exists(model_dir):
            os.mkdir(model_dir)
    return model_dir

def main(argv):
    if FLAGS.dataset.startswith("synthetic"):
        if FLAGS.method == "bt":
            train_answers, _, item_values, workers = data_utils.load_synthetic_dataset(FLAGS.data_root, FLAGS.dataset, "I", FLAGS.synthetic_part1_ratio, FLAGS.synthetic_noise_level, FLAGS.synthetic_round)
        elif FLAGS.method == "bt_II":
            train_answers, train_answers2, item_values, workers= data_utils.load_synthetic_dataset(FLAGS.data_root, FLAGS.dataset, "II", FLAGS.synthetic_part1_ratio, FLAGS.synthetic_noise_level, FLAGS.synthetic_round)
        else:
            raise NotImplementedError
    else:
        item_values, workers = data_utils.load_groundtruth(FLAGS.data_root, FLAGS.dataset)
        if FLAGS.method in ["sigmoidal_I", "bt", "crowd-bt_I", "neural_I", "mog_I", "hbtl_I", "hbtl", "moe_I"]:
            train_answers, _ = data_utils.load_dataset(FLAGS.data_root, FLAGS.dataset, "I", FLAGS.train_mode, subsampling=FLAGS.subsampling)
        elif FLAGS.method in ["sigmoidal_II", "bt_II", "crowd-bt_II", "neural_II", "mog_II", "hbtl_II", "moe_II"]:
            train_answers, train_answers2 = data_utils.load_dataset(FLAGS.data_root, FLAGS.dataset, "II", FLAGS.train_mode, subsampling=FLAGS.subsampling)
        else:
            raise NotImplementedError
    n_item, n_worker = len(item_values), len(workers)
    tf.reset_default_graph()
    step_t = tf.train.get_or_create_global_step()
    if FLAGS.decay_frequency == 0:
        lr_t = FLAGS.lr
    else:
        lr_t = tf.train.exponential_decay(FLAGS.lr, step_t, FLAGS.decay_frequency, 0.8)
    lr_t = tf.maximum(lr_t, 3e-4)

    params = train_utils.declare_params(n_worker, n_item, FLAGS.embedding_dim, FLAGS.method)
    place_holders = train_utils.declare_placeholders()
    batch_sampler = train_utils.BatchSamplerTypeI(FLAGS.batch_size, train_answers, place_holders)
    if FLAGS.method in ["sigmoidal_II", "bt_II", "crowd-bt_II", "neural_II", "mog_II", "hbtl_II", "moe_II"]:
        batch_sampler2 = train_utils.BatchSamplerTypeII(FLAGS.batch_size, train_answers2, place_holders)
    if FLAGS.method == "sigmoidal_I":
        algo_module = getattr(algorithms, "sigmoidal_utility_I")
    elif FLAGS.method == "sigmoidal_II":
        algo_module = getattr(algorithms, "sigmoidal_utility_II")
    elif FLAGS.method in ["bt", "bt_II"]:
        algo_module = getattr(algorithms, "bt")
    elif FLAGS.method == "crowd-bt_I":
        algo_module = getattr(algorithms, "crowd_bt_I")
    elif FLAGS.method == "crowd-bt_II":
        algo_module = getattr(algorithms, "crowd_bt_II")
    elif FLAGS.method == "neural_I":
        algo_module = getattr(algorithms, "neural_utility_I")
    elif FLAGS.method == "neural_II":
        algo_module = getattr(algorithms, "neural_utility_II")
    elif FLAGS.method == "mog_I":
        algo_module = getattr(algorithms, "mog_utility_I")
    elif FLAGS.method == "mog_II":
        algo_module = getattr(algorithms, "mog_utility_II")
    elif FLAGS.method == "hbtl_I":
        algo_module = getattr(algorithms, "hbtl_I")
    elif FLAGS.method == "hbtl_II":
        algo_module = getattr(algorithms, "hbtl_II")
    elif FLAGS.method == "moe_I":
        algo_module = getattr(algorithms, "moe_utility_I")
    elif FLAGS.method == "moe_II":
        algo_module = getattr(algorithms, "moe_utility_II")
    else:
        raise NotImplementedError
    train_step, loss, train_step2, loss2 = algo_module.create_train_loss_op(place_holders, params, FLAGS.l2_weight, lr_t, step_t)

    model_dir = create_model_dir(FLAGS.exp_name, FLAGS.method)
    ckpt_name = osp.join(model_dir, "model")
    saver = tf.train.Saver(max_to_keep=2)
    config = tf.ConfigProto()
    loss_val2 = 0
    if FLAGS.use_gpu:
        config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        if not tf.train.latest_checkpoint(model_dir) is None:
            saver.restore(sess, tf.train.latest_checkpoint(model_dir))
        else:
            sess.run(tf.global_variables_initializer())
        step = sess.run(step_t)
        while step < FLAGS.n_steps:
            feed_dict = batch_sampler.sample_batch()
            if FLAGS.method in ["sigmoidal_II", "bt_II", "neural_II", "crowd-bt_II", "mog_II", "hbtl_II", "moe_II"]:
                # need to sample from type II answers
                feed_dict.update(batch_sampler2.sample_batch())
            if FLAGS.method == "bt":
                # BT has one train op
                _, loss_val, lr_val, step = sess.run([train_step, loss, lr_t, step_t], feed_dict=feed_dict)
                if step % FLAGS.print_frequency == 0:
                    print(f"loss at step {step}: {loss_val}, lr: {lr_val}")
                    saver.save(sess, ckpt_name, global_step=step_t)
            else:
                _, loss_val1, lr_val, step = sess.run([train_step, loss, lr_t, step_t], feed_dict=feed_dict)
                _, loss_val2 = sess.run([train_step2, loss2], feed_dict=feed_dict)
                if step % FLAGS.print_frequency == 0:
                    print(f"step {step}, loss1: {loss_val1}, loss2: {loss_val2}, lr: {lr_val}")
                    saver.save(sess, ckpt_name, global_step=step_t)
        saver.save(sess, ckpt_name, global_step=step_t)
        c1 = sess.run(params["c1"])
        c2 = sess.run(params["c2"])
        item_emb = sess.run(params["item"])
    np.save(os.path.join(model_dir, "c1.npy"), c1)
    np.save(os.path.join(model_dir, "c2.npy"), c2)
    np.save(os.path.join(model_dir, "item_emb.npy"), item_emb)
if __name__ == '__main__':
    app.run(main)

    