import os.path as osp
import json

def load_dataset(data_root, dataset, query_type="I", split="train", subsampling="80"):
    data_file = osp.join(data_root, dataset, 'processed.json')
    with open(data_file, "r") as f:
        data_dict = json.loads(f.read())
    if split == "eval":
        return data_dict["answers_part4"], None
    elif split == "valid":
        return data_dict["answers_part3"], None
    elif split == "all":
        answers = data_dict["answers_part1"] + data_dict["answers_part2"] + data_dict["answers_part3"] + data_dict["answers_part4"]
        return answers, None
    if not split in ["train", "train_valid"]:
        raise NotImplementedError

    answers = data_dict["answers_part1"]
    if subsampling == "80":
        if query_type == "I":
            answers += data_dict["answers_part2"]
            answers_type2 = None
        else:
            answers_type2 = data_dict["answers_type2_flipped"]
    elif subsampling == "70":
        if query_type == "I":
            answers += data_dict['answers_part1_30']
            answers_type2 = None
        else:
            answers_type2 = data_dict['answers_type2_30']
    elif subsampling == "60":
        if query_type == "I":
            answers += data_dict['answers_part1_20']
            answers_type2 = None
        else:
            answers_type2 = data_dict['answers_type2_20']
    elif subsampling == "50":
        if query_type == "I":
            answers += data_dict['answers_part1_10']
            answers_type2 = None
        else:
            answers_type2 = data_dict['answers_type2_10']
    if split == "train_valid":
            answers += data_dict["answers_part3"]
    return answers, answers_type2 

def load_synthetic_dataset(data_root, dataset, query_type="I", part1_ratio=50, noise_level=0, round_=1):
    data_file = osp.join(data_root, dataset, f'processed.json')
    with open(data_file, "r") as f:
        data_dict = json.loads(f.read())
    workers = set()
    for old_id, new_id in data_dict["workerid_mapping"].items():
        if not new_id is None:
            workers.add(new_id)
    item_scores = data_dict["itemid_score"]
    print(data_dict.keys())
    data_dict = data_dict[str(noise_level)][str(part1_ratio)][str(round_)]
    answers = data_dict["answers_part1"]
    if query_type == "I":
        answers += data_dict["answers_part2"]
        answers_type2 = None
    else:
        answers_type2 = data_dict["answers_type2_flipped"]
    return answers, answers_type2, item_scores, workers   

def load_groundtruth(data_root, dataset):
    data_file = osp.join(data_root, dataset, 'processed.json')
    with open(data_file, "r") as f:
        data_dict = json.loads(f.read())
    true_scores = data_dict["itemid_score"]
    workers = set()
    for old_id, new_id in data_dict["workerid_mapping"].items():
        if not new_id is None:
            workers.add(new_id)
    return true_scores, workers

