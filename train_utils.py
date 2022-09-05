# coding: utf-8
import tensorflow.compat.v1 as tf
import numpy as np 

class BatchSamplerTypeI():
    def __init__(self, batch_size, answers, place_holders):
        self.batch_size = batch_size
        self.answers = answers
        self.place_holders = place_holders
    def sample_batch(self):
        if self.batch_size < len(self.answers):
            sample_ids = np.random.choice(len(self.answers), size=self.batch_size, replace=False)
        else:
            sample_ids = [i for i in range(len(self.answers))]
        item1, item2, wid, labels = [], [], [], []
        item1_reg_mask, item2_reg_mask = [], []
        worker_reg_mask = []
        item_counted, worker_counted = set(), set()
        # 在计算一个batch的正则项时，每个物品理应只算一次，使用两个mask实现这点
        # 否则出现频次高的物品的向量会被过度打压
        for ind in sample_ids:
            i1, i2, label, worker = self.answers[ind]
            for item, mask in zip([i1, i2], [item1_reg_mask, item2_reg_mask]):
                if item in item_counted:
                    mask.append(0.)
                else:
                    mask.append(1.)
                    item_counted.add(item)
            if worker in worker_counted:
                worker_reg_mask.append(0.)
            else:
                worker_reg_mask.append(1.)
                worker_counted.add(worker)
            for o,l in zip([i1, i2, label, worker], [item1, item2, labels, wid]):
                l.append(o)
        feed_dict = {self.place_holders["item1"]: item1,
                     self.place_holders["item2"]: item2,
                     self.place_holders["worker"]: wid,
                     self.place_holders["label"]: labels,
                     self.place_holders["item1_reg_mask"]: item1_reg_mask,
                     self.place_holders["item2_reg_mask"]: item2_reg_mask,
                     self.place_holders["worker_reg_mask"]: worker_reg_mask}
        return feed_dict

class BatchSamplerTypeII(BatchSamplerTypeI):
    def sample_batch(self):
        if self.batch_size < len(self.answers):
            sample_ids = np.random.choice(len(self.answers), size=self.batch_size, replace=False)
        else:
            sample_ids = [i for i in range(len(self.answers))]
        item11, item12, label1, item21, item22, label2 = [], [], [], [], [], []
        wid, label_type2 = [], []
        item11_reg_mask, item12_reg_mask = [], []
        item21_reg_mask, item22_reg_mask = [], []
        worker_reg_mask = []
        item_counted, worker_counted = set(), set()
        for ind in sample_ids:
            answer = self.answers[ind]
            i11, i12, l1 = answer[0], answer[1], answer[2]
            i21, i22, l2 = answer[3], answer[4], answer[5]
            l_typ2, worker = answer[6], float(answer[7])

            for item, mask in zip([i11, i12, i21, i22],
                                  [item11_reg_mask, item12_reg_mask, item21_reg_mask, item22_reg_mask]):
                if item in item_counted:
                    mask.append(0.)
                else:
                    mask.append(1.)
                    item_counted.add(item)
            if worker in worker_counted:
                worker_reg_mask.append(0.)
            else:
                worker_reg_mask.append(1.)
                worker_counted.add(worker)
            for o, l in zip([i11, i12, l1, i21, i22, l2, l_typ2, worker],
            [item11, item12, label1, item21, item22, label2, label_type2, wid]):
                l.append(o)
        feed_dict = {self.place_holders["item11"]: item11,
                     self.place_holders["item12"]: item12,
                     self.place_holders["label1"]: label1,
                     self.place_holders["item21"]: item21,
                     self.place_holders["item22"]: item22,
                     self.place_holders["label2"]: label2,
                     self.place_holders["label_type2"]: label_type2,
                     self.place_holders["worker_II"]: wid,
                     self.place_holders["item11_reg_mask"]: item11_reg_mask,
                     self.place_holders["item12_reg_mask"]: item12_reg_mask,
                     self.place_holders["item21_reg_mask"]: item21_reg_mask,
                     self.place_holders["item22_reg_mask"]: item22_reg_mask,
                     self.place_holders["worker_II_reg_mask"]: worker_reg_mask}
        return feed_dict

def declare_params(n_worker, n_item, embedding_dim, method=None):
    with tf.variable_scope("worker_param", reuse=tf.AUTO_REUSE):
        if method in ["crowd-bt_I", "crowd-bt_II"]:
            ini =  5 * np.ones(((n_worker, embedding_dim)), dtype=np.float32)
            c1 = tf.get_variable("c1", initializer=ini)
            c2 = tf.get_variable("c2", (n_worker, embedding_dim))
            item = tf.get_variable("item", (n_item, embedding_dim))
        elif method in ["hbtl", "hbtl_I", "hbtl_II"]:
            ini = np.ones(((n_worker, embedding_dim)), dtype=np.float32)
            c1 = tf.get_variable("c1", initializer=ini)
            c2 = tf.get_variable("c2", (n_worker, embedding_dim))
            ini = np.ones(((n_item, embedding_dim)), dtype=np.float32)
            item = tf.get_variable("item", initializer=ini)
        elif method in ["bt", "bt_II"]:
            c1 = tf.get_variable("c1", (n_worker, embedding_dim))
            c2 = tf.get_variable("c2", (n_worker, embedding_dim))
            item = tf.get_variable("item", (n_item, embedding_dim))
        elif method in ["sigmoidal_I", "sigmoidal_II"]:
            ini = np.ones(((n_worker, embedding_dim)), dtype=np.float32)
            c1 = tf.get_variable("c1", initializer=ini)
            ini = np.zeros(((n_worker, embedding_dim)), dtype=np.float32)
            c2 = tf.get_variable("c2", initializer=ini)
            #item = tf.get_variable("item", (n_item, embedding_dim))
            ini = np.ones(((n_item, embedding_dim)), dtype=np.float32)
            item = tf.get_variable("item", initializer=ini)
        elif method in ["mog_I", "mog_II", "moe_I", "moe_II"]:
            ini = np.ones(((n_worker, 4)), dtype=np.float32) / 4.
            c1 = tf.get_variable("c1", initializer=ini)
            ini = np.concatenate(
                    [np.random.random(((4, 1))).astype(np.float32)-0.5, np.ones(((4, 1)), dtype=np.float32)],
                    axis=1)
            c2 = tf.get_variable("c2", initializer=ini)
            #item = tf.get_variable("item", (n_item, embedding_dim))
            ini = np.ones(((n_item, embedding_dim)), dtype=np.float32)
            item = tf.get_variable("item", initializer=ini)
        elif method in ["neural_I", "neural_II"]:
            bias_init = np.zeros((n_worker, 1, 4))
            weight_init = np.random.random((embedding_dim, 4)) - 0.5
            weight_init = np.array([weight_init] * n_worker)
            ini = np.concatenate([weight_init, bias_init], axis=1).astype(np.float32)
            #c1 = tf.get_variable("c1", (n_worker, embedding_dim+1, 4))
            c1 = tf.get_variable("c1", initializer=ini)
            bias_init = np.array([0.] * n_worker).astype(np.float32).reshape((n_worker, 1))
            weight_init = np.random.random(4)
            weight_init = np.array([weight_init] * n_worker)
            ini = np.concatenate([weight_init, bias_init], axis=1).astype(np.float32)
            #c2 = tf.get_variable("c2", (n_worker, 5))
            c2 = tf.get_variable("c2", initializer=ini)
            #item = tf.get_variable("item", (n_item, embedding_dim))
            ini = np.ones(((n_item, embedding_dim)), dtype=np.float32)
            item = tf.get_variable("item", initializer=ini)
        else:
            raise NotImplementedError    
    params = dict(c1=c1, c2=c2, item=item)
    return params

def declare_placeholders():
    d = declare_placeholders_for_typeI()
    d.update(declare_placeholders_typeII())
    return d
def declare_placeholders_for_typeI():
    item1 = tf.placeholder(tf.int64, shape=(None, ))
    item2 = tf.placeholder(tf.int64, shape=(None, ))
    worker = tf.placeholder(tf.int64, shape=(None, ))
    label = tf.placeholder(tf.float32, shape=(None, ))
    item1_reg_mask = tf.placeholder(tf.float32, shape=(None, ))
    item2_reg_mask = tf.placeholder(tf.float32, shape=(None, ))
    worker_reg_mask = tf.placeholder(tf.float32, shape=(None, ))
    place_holders = dict(item1=item1, item2=item2,
                         worker=worker, label=label,
                         item1_reg_mask=item1_reg_mask,
                         item2_reg_mask=item2_reg_mask,
                         worker_reg_mask=worker_reg_mask)
    return place_holders

def declare_placeholders_typeII():
    item11 = tf.placeholder(tf.int64, shape=(None, ))
    item12 = tf.placeholder(tf.int64, shape=(None, ))
    item21 = tf.placeholder(tf.int64, shape=(None, ))
    item22 = tf.placeholder(tf.int64, shape=(None, ))
    worker = tf.placeholder(tf.int64, shape=(None, ))
    label1 = tf.placeholder(tf.float32, shape=(None, ))
    label2 = tf.placeholder(tf.float32, shape=(None, ))
    label_type2 = tf.placeholder(tf.float32, shape=(None, ))
    item11_reg_mask = tf.placeholder(tf.float32, shape=(None, ))
    item12_reg_mask = tf.placeholder(tf.float32, shape=(None, ))
    item21_reg_mask = tf.placeholder(tf.float32, shape=(None, ))
    item22_reg_mask = tf.placeholder(tf.float32, shape=(None, ))
    worker_reg_mask = tf.placeholder(tf.float32, shape=(None, ))
    place_holders = dict(item11=item11, item12=item12, 
                         item21=item21, item22=item22,
                         worker_II=worker, 
                         label1=label1,
                         label2=label2, label_type2=label_type2,
                         item11_reg_mask=item11_reg_mask,
                         item12_reg_mask=item12_reg_mask,
                         item21_reg_mask=item21_reg_mask,
                         item22_reg_mask=item22_reg_mask,
                         worker_II_reg_mask=worker_reg_mask
                         )
    return place_holders
