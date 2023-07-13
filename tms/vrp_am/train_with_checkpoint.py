import tensorflow as tf
from time import gmtime, strftime

from tms.vrp_am.attention_model import AttentionModel, set_decode_type
from tms.vrp_am.reinforce_baseline import RolloutBaseline
from tms.vrp_am.train import train_model

from tms.vrp_am.util import create_data_on_disk, get_cur_time

# Params of model
SAMPLES = 128000  # 512*250
BATCH = 512
START_EPOCH = 0
END_EPOCH = 5
FROM_CHECKPOINT = False
embedding_dim = 128
LEARNING_RATE = 0.0001
ROLLOUT_SAMPLES = 10000
NUMBER_OF_WP_EPOCHS = 1
GRAD_NORM_CLIPPING = 1.0
BATCH_VERBOSE = 1000
VAL_BATCH_SIZE = 1000
VALIDATE_SET_SIZE = 10000
SEED = 1234
GRAPH_SIZE = 20
FILENAME = 'VRP_{}_{}'.format(GRAPH_SIZE, strftime("%Y-%m-%d", gmtime()))

# Initialize model
model_tf = AttentionModel(embedding_dim)
set_decode_type(model_tf, "sampling")
print(get_cur_time(), 'model initialized')

# Create and save validation dataset
validation_dataset = create_data_on_disk(GRAPH_SIZE,
                                         VALIDATE_SET_SIZE,
                                         is_save=True,
                                         filename=FILENAME,
                                         is_return=True,
                                         seed=SEED)
print(get_cur_time(), 'validation dataset created and saved on the disk')

# Initialize optimizer
optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

# Initialize baseline
baseline = RolloutBaseline(model_tf,
                           wp_n_epochs=NUMBER_OF_WP_EPOCHS,
                           epoch=0,
                           num_samples=ROLLOUT_SAMPLES,
                           filename=FILENAME,
                           from_checkpoint=FROM_CHECKPOINT,
                           embedding_dim=embedding_dim,
                           graph_size=GRAPH_SIZE
                           )
print(get_cur_time(), 'baseline initialized')

train_model(optimizer,
            model_tf,
            baseline,
            validation_dataset,
            samples=SAMPLES,
            batch=BATCH,
            val_batch_size=VAL_BATCH_SIZE,
            start_epoch=START_EPOCH,
            end_epoch=END_EPOCH,
            from_checkpoint=FROM_CHECKPOINT,
            grad_norm_clipping=GRAD_NORM_CLIPPING,
            batch_verbose=BATCH_VERBOSE,
            graph_size=GRAPH_SIZE,
            filename=FILENAME
            )


# # ######################################
# import tensorflow as tf
# from time import gmtime, strftime
#
# from tms.vrp_am.attention_model import set_decode_type
# from tms.vrp_am.reinforce_baseline import RolloutBaseline
# from tms.vrp_am.train import train_model
#
# from tms.vrp_am.util import get_cur_time
# from tms.vrp_am.reinforce_baseline import load_tf_model
# from tms.vrp_am.util import read_from_pickle
#
# SAMPLES = 128000  # 512*250
# BATCH = 512
# LEARNING_RATE = 0.0001
# ROLLOUT_SAMPLES = 10000
# NUMBER_OF_WP_EPOCHS = 1
# GRAD_NORM_CLIPPING = 1.0
# BATCH_VERBOSE = 1000
# VAL_BATCH_SIZE = 1000
# VALIDATE_SET_SIZE = 10000
# SEED = 1234
# GRAPH_SIZE = 20
# FILENAME = 'VRP_{}_{}'.format(GRAPH_SIZE, strftime("%Y-%m-%d", gmtime()))
#
# START_EPOCH = 5
# END_EPOCH = 10
# FROM_CHECKPOINT = True
# embedding_dim = 128
# MODEL_PATH = 'model_checkpoint_epoch_4_VRP_20_2023-07-13.h5'
# VAL_SET_PATH = 'Validation_dataset_VRP_20_2023-07-13.pkl'
# BASELINE_MODEL_PATH = 'baseline_checkpoint_epoch_4_VRP_20_2023-07-13.h5'
#
# # Initialize model
# model_tf = load_tf_model(MODEL_PATH,
#                          embedding_dim=embedding_dim,
#                          graph_size=GRAPH_SIZE)
# set_decode_type(model_tf, "sampling")
# print(get_cur_time(), 'model loaded')
#
# # Create and save validation dataset
# validation_dataset = read_from_pickle(VAL_SET_PATH)
# print(get_cur_time(), 'validation dataset loaded')
#
# # Initialize optimizer
# optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
#
# # Initialize baseline
# baseline = RolloutBaseline(model_tf,
#                            wp_n_epochs=NUMBER_OF_WP_EPOCHS,
#                            epoch=START_EPOCH,
#                            num_samples=ROLLOUT_SAMPLES,
#                            filename=FILENAME,
#                            from_checkpoint=FROM_CHECKPOINT,
#                            embedding_dim=embedding_dim,
#                            graph_size=GRAPH_SIZE,
#                            path_to_checkpoint=BASELINE_MODEL_PATH)
# print(get_cur_time(), 'baseline initialized')
#
# train_model(optimizer,
#             model_tf,
#             baseline,
#             validation_dataset,
#             samples=SAMPLES,
#             batch=BATCH,
#             val_batch_size=VAL_BATCH_SIZE,
#             start_epoch=START_EPOCH,
#             end_epoch=END_EPOCH,
#             from_checkpoint=FROM_CHECKPOINT,
#             grad_norm_clipping=GRAD_NORM_CLIPPING,
#             batch_verbose=BATCH_VERBOSE,
#             graph_size=GRAPH_SIZE,
#             filename=FILENAME
#             )
