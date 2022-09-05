import numpy as np
import os.path as osp
import os
from sklearn.metrics import ndcg_score, log_loss, roc_auc_score
import scipy
from sklearn.utils.extmath import softmax

def sigmoid(x):
    return 1. / (1. + np.exp(-x))
def log_sigmoid(x):
    return np.log(sigmoid(x))
def softplus(x):
    return np.log1p(np.exp(x))
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1)[:, None]
def log_softmax(x):
    return np.log(softmax(x))
def relu(x):
    flag = (x > 0).astype(np.float32)
    return flag * x + (1-flag) * np.zeros_like(x)

def predict_bt(answers, c1s, c2s, item_embs):
    predictions = []
    for ans in answers:
        prob = sigmoid(np.sum(item_embs[ans[0]]-item_embs[ans[1]]))
        predictions.append(prob)
    return np.array(predictions)

def predict_crowdbt(answers, c1s, c2s, item_embs):
    predictions = []
    log_p_bt_matrix = log_sigmoid(item_embs - item_embs.T)
    log_reliability_vec, log_reliability_vec_ = log_sigmoid(c1s), log_sigmoid(-c1s)
    for ans in answers:
        log_p_bt, log_p_bt_= log_p_bt_matrix[ans[0]][ans[1]], log_p_bt_matrix[ans[1]][ans[0]] 
        log_reliability = log_reliability_vec[ans[3]][0]
        log_reliability_ = log_reliability_vec_[ans[3]][0]
        prob12= np.exp(log_p_bt + log_reliability)+np.exp(log_p_bt_ + log_reliability_)
        predictions.append(prob12)
    return np.array(predictions)

def compute_score_matrix(c1s, c2s, item_embs, method):
    if method == "sigmoidal":
        return sigmoid(np.sum((item_embs[:, None, :] - c2s[None, :, :]) * c1s[None, :, :], axis=2))
    elif method == "hbtl":
        return np.exp(np.sum(item_embs[:, None, :] * c1s[None, :, :], axis=2))
    elif method == "mog":
        weight_logits = c1s
        mu, sigma = c2s[:, 0], softplus(c2s[:, 1])
        # n_item, 1 - None,  4 -> n_item, 4
        components = np.exp(-(item_embs - mu[None, :])**2 / (2. * sigma[None, :]**2))
        # n_worker, 4
        weights = softmax(c1s)
        return np.sum(weights[None, :, :] * components[:, None, :], axis=2)
    elif method == "moe":
        weight_logits = c1s
        mu, sigma = c2s[:, 0], c2s[:, 1]
        # n_item, 1 - None,  4 -> n_item, 4
        components = np.exp((item_embs - mu[None, :]) * sigma[None, :])
        # n_worker, 4
        weights = softmax(c1s)
        return np.sum(weights[None, :, :] * components[:, None, :], axis=2)
    elif method == "neural":
        # n_worker, embedding, k; n_worker, k
        w1, b1 = c1s[:, :-1], c1s[:, -1]
        # n_worker, k; n_worker
        w2, b2 = c2s[:, :-1], c2s[:, -1]
        #(n_item, n_worker, embedding, 2) * (n_item, n_worker, embedding_dim, 2)
        item_scores = np.sum(item_embs[:, None, :, None] * w1[None, :, :], axis=2) + b1[None, :, :]
        # n_item, n_worker, 2
        return sigmoid(np.sum(sigmoid(item_scores) * w2[None,:, :], axis=2) + b2[None, :])
        #return relu( np.sum(relu(item_scores) * w2[None,:, :], axis=2) + b2[None, :])
    else:
        raise NotImplementedError

def predict_answer_from_score_matrix(answers, matrix):
    predictions = []  
    for ans in answers:
        util1 = matrix[ans[0]][ans[-1]]
        util2 = matrix[ans[1]][ans[-1]]
        if util1 + util2 == 0:
            prob = 0.
        else:
            prob = (util1 / (util1 + util2))
        predictions.append(prob)
    return np.array(predictions)

def predict_order_from_score_matrix(answers, matrix):
    matrix = np.mean(matrix, axis=1)
    predictions = []
    for ans in answers:
        util1 = matrix[ans[0]]
        util2 = matrix[ans[1]]
        if util1 + util2 == 0:
            prob = 0.
        else:
            prob = (util1 / (util1 + util2))
        predictions.append(prob)
    return np.array(predictions)

def compute_gauc(answers, predictions, item_scores=None):
    d = {}
    for ind, ans in enumerate(answers):
        worker = ans[3]
        if not worker in d:
            d[worker] = ([], [])
        d[worker][0].append(predictions[ind])
        if item_scores is None:
            d[worker][1].append(ans[2])
        else:
            s1 = item_scores[str(ans[0])]
            s2 = item_scores[str(ans[1])]
            pred = predictions[ind]
            if (s1 > s2 and pred > 0.5) or (s1 == s2 and pred == 0.5) or (s1 < s2 and pred < 0.5):
                d[worker][1].append(1.)
            else:
                 d[worker][1].append(0.)
    gauc = {}
    for worker, (preds, labels) in d.items():
        # filter label=0.5
        p, l = [], []
        for i in range(len(labels)):
            if labels[i] == 0.5:
                continue
            else:
                p.append(preds[i])
                l.append(labels[i])
        if sum(l) == 0 or sum(l) == len(l):
            continue
        auc = roc_auc_score(l, p)
        gauc[worker] = (auc, len(l))
    sum_count, sum_auc = 0, 0
    for _, (value, count) in gauc.items():
        sum_count += count
        sum_auc += value * count
    return sum_auc / sum_count

def compute_acc(answers, predictions, item_scores=None):
    acc = 0
    for ind, ans in enumerate(answers):
        if predictions[ind] < 0.5:
            pred = 0
        elif predictions[ind] == 0.5:
            pred = 0.5
        else:
            pred = 1
        if item_scores is None:
            if pred == ans[2]:
                acc += 1
        else:
            s1 = item_scores[str(ans[0])]
            s2 = item_scores[str(ans[1])]
            if (s1 > s2 and pred == 1) or (s1 == s2 and pred == 0.5) or (s1 < s2 and pred == 0.):
                acc += 1
    return acc / len(predictions)

def hit_rate( true_values, prediction, top_k):
    return len(set(np.argsort(prediction)[-top_k:]) & set(np.argsort(true_values)[-top_k:])) / top_k

def evaluate_answer_prediction(exp_path, answers, method, item_scores):
    labels = np.array([i[2] for i in answers])
    item_emb = np.load(osp.join(exp_path, method, "item_emb.npy"))
    c1 = np.load(osp.join(exp_path, method, "c1.npy"))
    c2 = np.load(osp.join(exp_path, method, "c2.npy"))
    if method in ["sigmoidal_I", "sigmoidal_II"]:
        score_matrix = compute_score_matrix(c1, c2, item_emb, "sigmoidal")
        predicted_answers = predict_answer_from_score_matrix(answers, score_matrix)
    elif method in ["hbtl_I", "hbtl_II"]:
        score_matrix = compute_score_matrix(c1, c2, item_emb, "hbtl")
        predicted_answers = predict_answer_from_score_matrix(answers, score_matrix)
    elif method in ["bt", "bt_II"]:
        predicted_answers = predict_bt(answers, None, None, item_emb)
    elif method in ["crowd-bt_I", "crowd-bt_II"]:
        predicted_answers = predict_crowdbt(answers, c1, None, item_emb)
    elif method in ["mog_I", "mog_II"]:
        score_matrix = compute_score_matrix(c1, c2, item_emb, "mog")
        predicted_answers = predict_answer_from_score_matrix(answers, score_matrix)
    elif method in ["moe_I", "moe_II"]:
        score_matrix = compute_score_matrix(c1, c2, item_emb, "moe")
        predicted_answers = predict_answer_from_score_matrix(answers, score_matrix)
    elif method in ["neural_I", "neural_II"]:
        score_matrix = compute_score_matrix(c1, c2, item_emb, "neural")
        predicted_answers = predict_answer_from_score_matrix(answers, score_matrix)
    auc = roc_auc_score(labels[labels!=0.5], predicted_answers[labels!=0.5])
    gauc = compute_gauc(answers, predicted_answers)
    acc = compute_acc(answers, predicted_answers)
    if method in ["sigmoidal_I", "sigmoidal_II", "hbtl_I", "hbtl_II", "mog_I", "mog_II", "neural_I", "neural_II", "moe_I", "moe_II"]:
        predictions = predict_order_from_score_matrix(answers, score_matrix)
    else:
        predictions = predicted_answers
    acc2 = compute_acc(answers, predictions, item_scores)
    gauc2 = compute_gauc(answers, predictions, item_scores)
    return {"auc": auc, "gauc": gauc, "acc": acc, "acc2": acc2, "gauc2": gauc2}

def compute_item_scores(exp_path, method):
    item_emb = np.load(osp.join(exp_path, method, "item_emb.npy"))
    c1 = np.load(osp.join(exp_path, method, "c1.npy"))
    c2 = np.load(osp.join(exp_path, method, "c2.npy"))
    if method in ["sigmoidal_I", "sigmoidal_II","mog_I", "mog_II", "neural_I", "neural_II","moe_I", "moe_II"]:
        score_matrix = compute_score_matrix(c1, c2, item_emb, method.split("_")[0])
    elif method in ["bt", "bt_II", "crowd-bt_I", "crowd-bt_II", "hbtl_I", "hbtl_II"]:
        score_matrix= item_emb
    return score_matrix