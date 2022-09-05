import os.path as osp
import os
import subprocess

data2steps = {}
data2steps[f"synthetic"] = 5000
data2steps[f"synthetic_fixed"] = 5000
#for dataname in ["crowd-bt", "olympic", "bike", "cheat", "night", "imdb-wiki-sbs", ]: # cheat-new
for round_ in range(6, 10):
    dataname = f"synthetic"
    for noise_level in [0., 0.1, 0.2, 0.3]:
        for part1_ratio in [50, 60, 70, 80, 90]:
            for exp_id in range(5,6):
                for method in ["bt", "bt_II"]:
                    if method == "bt" and not part1_ratio == 50:
                        continue
                    exp_name = f"{dataname}-{noise_level}-{part1_ratio}-{round_}-{exp_id}"
                    train_str = f"python train.py --dataset {dataname} --n_steps {data2steps[dataname]} --train_mode train_valid --method {method} --exp_name {exp_name}"
                    #if data2steps[dataname] == 3000 and not method.startswith("crowd-bt"):
                    # --print_frequency 100
                    train_str += " --decay_frequency 200"
                    train_str += f" --synthetic_noise_level {noise_level}"
                    train_str += f" --synthetic_part1_ratio {part1_ratio}"
                    train_str += f" --synthetic_round {round_}"
                    print(round_, noise_level, part1_ratio, exp_id, method)
                    print("******")
                    os.makedirs(osp.join("experiments", exp_name, method), exist_ok=True)
                    output_file = osp.join("experiments",  exp_name, method, "train_log.txt")
                    os.system(train_str + " > " + output_file)
                    p = subprocess.run([train_str], check=True, shell=True, capture_output=True)
                    with open(osp.join("experiments",  exp_name, method, "train_log.txt"), "w") as f:
                        f.write(p.stderr.decode())