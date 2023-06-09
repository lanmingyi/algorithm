import tensorflow as tf
from scipy.stats import ttest_rel
from tqdm import tqdm
import numpy as np

from attention_dynamic_model import AttentionDynamicModel
from attention_dynamic_model import set_decode_type
from utils import generate_data_onfly


def copy_of_tf_model(model, embedding_dim=128, graph_size=20):
    """Copy model weights to new model
    """
    # https://stackoverflow.com/questions/56841736/how-to-copy-a-network-in-tensorflow-2-0
    CAPACITIES = {10: 20.,
                  20: 30.,
                  50: 40.,
                  100: 50.
                  }

    data_random = [tf.random.uniform((2, 2,), minval=0, maxval=1, dtype=tf.dtypes.float32),
                   tf.random.uniform((2, graph_size, 2), minval=0, maxval=1, dtype=tf.dtypes.float32),
                   tf.cast(tf.random.uniform(minval=1, maxval=10, shape=(2, graph_size),
                                             dtype=tf.int32), tf.float32) / tf.cast(CAPACITIES[graph_size], tf.float32)]

    new_model = AttentionDynamicModel(embedding_dim)
    set_decode_type(new_model, "sampling")
    _, _ = new_model(data_random)

    for a, b in zip(new_model.variables, model.variables):
        a.assign(b)

    return new_model


def rollout(model, dataset, batch_size=1000, disable_tqdm=False):
    # Evaluate model in greedy mode 在贪心模式下计算模型
    set_decode_type(model, "greedy")
    costs_list = []

    for batch in tqdm(dataset.batch(batch_size), disable=disable_tqdm, desc="Rollout greedy execution"):
        cost, _ = model(batch)
        costs_list.append(cost)

    return tf.concat(costs_list, axis=0)


def validate(dataset, model, batch_size=1000):
    """Validates model on given dataset in greedy mode
    """
    val_costs = rollout(model, dataset, batch_size=batch_size)
    set_decode_type(model, "sampling")
    mean_cost = tf.reduce_mean(val_costs)
    print(f"Validation score: {np.round(mean_cost, 4)}")
    return mean_cost


class RolloutBaseline:

    def __init__(self, model, filename,
                 from_checkpoint=False,
                 path_to_checkpoint=None,
                 wp_n_epochs=1,
                 epoch=0,
                 num_samples=10000,
                 warmup_exp_beta=0.8,
                 embedding_dim=128,
                 graph_size=20
                 ):
        """
        Args:
            model: current model
            filename: suffix for baseline checkpoint filename
            from_checkpoint: start from checkpoint flag
            path_to_checkpoint: path to baseline model weights
            wp_n_epochs: number of warm-up epochs
            epoch: current epoch number
            num_samples: number of samples to be generated for baseline dataset
            warmup_exp_beta: warmup mixing parameter (exp. moving average parameter)

        """

        self.num_samples = num_samples
        self.cur_epoch = epoch
        self.wp_n_epochs = wp_n_epochs
        self.beta = warmup_exp_beta

        # controls the amount of warmup
        self.alpha = 0.0

        self.running_average_cost = None

        # Checkpoint params
        self.filename = filename
        self.from_checkpoint = from_checkpoint
        self.path_to_checkpoint = path_to_checkpoint

        # Problem params
        self.embedding_dim = embedding_dim
        self.graph_size = graph_size

        # create and evaluate initial baseline
        self._update_baseline(model, epoch)

    def _update_baseline(self, model, epoch):

        # Load or copy baseline model based on self.from_checkpoint condition
        if self.from_checkpoint and self.alpha == 0:
            print('Baseline model loaded')
            self.model = load_tf_model(self.path_to_checkpoint,
                                       embedding_dim=self.embedding_dim,
                                       graph_size=self.graph_size)
        else:
            self.model = copy_of_tf_model(model,
                                          embedding_dim=self.embedding_dim,
                                          graph_size=self.graph_size)

            # For checkpoint
            self.model.save_weights('baseline_checkpoint_epoch_{}_{}.h5'.format(epoch, self.filename), save_format='h5')

        # We generate a new dataset for baseline model on each baseline update to prevent possible overfitting
        self.dataset = generate_data_onfly(num_samples=self.num_samples, graph_size=self.graph_size)

        print(f"Evaluating baseline model on baseline dataset (epoch = {epoch})")
        self.bl_vals = rollout(self.model, self.dataset)
        self.mean = tf.reduce_mean(self.bl_vals)
        self.cur_epoch = epoch

    def ema_eval(self, cost):
        """This is running average of cost through previous batches (only for warm-up epochs)
            这是之前批次的运行平均成本(仅用于预热阶段)
        """

        if self.running_average_cost is None:
            self.running_average_cost = tf.reduce_mean(cost)
        else:
            self.running_average_cost = self.beta * self.running_average_cost + (1. - self.beta) * tf.reduce_mean(cost)

        return self.running_average_cost

    def eval(self, batch, cost):
        """Evaluates current baseline model on single training batch
            在单个训练批次上评估当前的基线模型
        """

        if self.alpha == 0:
            return self.ema_eval(cost)

        if self.alpha < 1:
            v_ema = self.ema_eval(cost)
        else:
            v_ema = 0.0

        v_b, _ = self.model(batch)

        v_b = tf.stop_gradient(v_b)
        v_ema = tf.stop_gradient(v_ema)

        # Combination of baseline cost and exp. moving average cost
        # 基线成本和经验的组合。移动平均成本
        return self.alpha * v_b + (1 - self.alpha) * v_ema

    def eval_all(self, dataset):
        """Evaluates current baseline model on the whole dataset only for non warm-up epochs
            仅在非暖化时期评估整个数据集上的当前基线模型
        """

        if self.alpha < 1:
            return None

        val_costs = rollout(self.model, dataset, batch_size=2048)

        return val_costs

    def epoch_callback(self, model, epoch):
        """Compares current baseline model with the training model and updates baseline if it is improved
            将当前基线模型与训练模型进行比较，如果基线有所改进，则更新基线
        """

        self.cur_epoch = epoch
        # Evaluating candidate model on baseline dataset
        print(f"Evaluating candidate model on baseline dataset (callback epoch = {self.cur_epoch})")
        candidate_vals = rollout(model, self.dataset)  # costs for training model on baseline dataset
        candidate_mean = tf.reduce_mean(candidate_vals)

        diff = candidate_mean - self.mean

        print(
            f"Epoch {self.cur_epoch} candidate mean {candidate_mean}, baseline epoch {self.cur_epoch} mean {self.mean}, difference {diff}")

        if diff < 0:
            # statistic + p-value
            t, p = ttest_rel(candidate_vals, self.bl_vals)

            p_val = p / 2
            print(f"p-value: {p_val}")

            if p_val < 0.05:
                print('Update baseline')
                self._update_baseline(model, self.cur_epoch)

        # alpha controls the amount of warmup Alpha控制预热学习率的量
        if self.alpha < 1.0:
            self.alpha = (self.cur_epoch + 1) / float(self.wp_n_epochs)
            print(f"alpha was updated to {self.alpha}")


def load_tf_model(path, embedding_dim=128, graph_size=20, n_encode_layers=2):
    """Load model weights from hd5 file 从hd5文件加载模型权重
    """
    # https://stackoverflow.com/questions/51806852/cant-save-custom-subclassed-model
    CAPACITIES = {10: 20.,
                  20: 30.,
                  50: 40.,
                  100: 50.
                  }

    data_random = [tf.random.uniform((2, 2,), minval=0, maxval=1, dtype=tf.dtypes.float32),
                   tf.random.uniform((2, graph_size, 2), minval=0, maxval=1, dtype=tf.dtypes.float32),
                   tf.cast(tf.random.uniform(minval=1, maxval=10, shape=(2, graph_size),
                                             dtype=tf.int32), tf.float32) / tf.cast(CAPACITIES[graph_size], tf.float32)]

    model_loaded = AttentionDynamicModel(embedding_dim, n_encode_layers=n_encode_layers)
    set_decode_type(model_loaded, "greedy")
    _, _ = model_loaded(data_random)

    model_loaded.load_weights(path)

    return model_loaded
