import os.path as osp
import os


data2steps = {"imdb-wiki-sbs": 10000, "bike-new": 3000, "cheat-new": 3000}
for d in ["ai_character", "olympic", "bike", "cheat","meeting", "night", "visitor", "crowd-bt"]:
    data2steps[d] = 5000
#for dataname in ["crowd-bt", "olympic", "bike", "cheat", "night", "imdb-wiki-sbs", ]: # cheat-new
for dataname in ["imdb-wiki-sbs", "crowd-bt", "olympic", "bike", "cheat",  "night", "ai_character", "visitor", "cheat-new", "bike-new"]:
    for exp_id in range(5,6):
        for method in [ "neural_I", "neural_II", "moe_I", "moe_II",
                        "bt", "bt_II", "crowd-bt_I", "crowd-bt_II", "hbtl_I", "hbtl_II"]:
        #for method in ["moe_I", "moe_II"]:
            train_str = f"python train.py --dataset {dataname} --n_steps {data2steps[dataname]} --train_mode train_valid --method {method} --exp_name {dataname}-{exp_id}"
            if method.startswith("crowd-bt"):
                train_str += " --lr 1e-3  --decay_frequency 0"
            if data2steps[dataname] == 3000 and not method.startswith("crowd-bt"):
                train_str += " --decay_frequency 500"
            if (method.startswith("moe") or method.startswith("mog")) and dataname.endswith("new"):
                train_str += " --l2_weight 0"
            print(dataname, method, exp_id)
            if not osp.exists(osp.join("experiments", f"{dataname}-{exp_id}")):
                os.mkdir(osp.join("experiments", f"{dataname}-{exp_id}"))
            if not osp.exists(osp.join("experiments", f"{dataname}-{exp_id}", method)):
                os.mkdir(osp.join("experiments", f"{dataname}-{exp_id}", method))
            output_file = osp.join("experiments",  f"{dataname}-{exp_id}", method, "train_log.txt")
            os.system(train_str + " > " + output_file)