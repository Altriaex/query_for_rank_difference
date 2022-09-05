from absl import app, flags

flags.DEFINE_string("data_root", "datasets", "the root directory of all datasets")
flags.DEFINE_string("dataset", "imdb-wiki-sbs", "the dataset to use")
flags.DEFINE_string("subsampling", "80", "ratio of train data to use")
flags.DEFINE_float("lr", 5e-4, "the learning rate")
flags.DEFINE_float("decay_frequency", 2000, "the learning rate")
flags.DEFINE_integer("batch_size", 256, "the size of batch")
flags.DEFINE_integer("print_frequency", 1000, "the frequence to print loss")
flags.DEFINE_integer("embedding_dim", 1, "the size of embedding")
flags.DEFINE_float("l2_weight", 1e-3, "the weight of l2 regularization")
flags.DEFINE_integer("n_steps", 50000, "the number of gradient steps")
flags.DEFINE_enum("method", "sigmoidal_I", ["sigmoidal_I", "sigmoidal_II", "mog_I", "bt", "crowd-bt_I", "crowd-bt_II",
                                             "bt_II",  "neural_I", "neural_II", "mog_II", "hbtl_I", "hbtl_II", "moe_I", "moe_II"], "the method to train")
flags.DEFINE_enum("train_mode", "train", ["train", "train_valid"], "whether include validation data in training")
flags.DEFINE_string("exp_name", None, "the name of experiment")
flags.DEFINE_boolean("use_gpu", True, "whether to use gpu")
flags.DEFINE_float('synthetic_noise_level', 0, '')
flags.DEFINE_integer('synthetic_part1_ratio', 50, '')
flags.DEFINE_integer('synthetic_round', 1, '')
FLAGS = flags.FLAGS
flags.mark_flag_as_required("exp_name")