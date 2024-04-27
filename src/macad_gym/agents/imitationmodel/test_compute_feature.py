import numpy as np
import tensorflow as tf
from macad_gym.agents.imitationmodel.imitation_learning import ImitationLearning
import time
import random
if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    sess = tf.Session(config = config)

    Image_agent = ImitationLearning(sess)
    Vec_ = []
    Image_agent.load_model()

    data = array = np.random.random((88, 200, 3)) * 255
    feature_vec = Image_agent.compute_feature(data)
    Vec_.append(feature_vec)
    print("feature:",feature_vec.shape, feature_vec.shape[0])
    data = np.array(feature_vec)
    print("data shape:",data.shape)
    V = np.array(Vec_)
    # print('max', V.max(axis=0))
    # print('min', V.min(axis=0))
    # print('mean', V.mean(axis=0))
