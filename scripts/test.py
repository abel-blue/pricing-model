# uncomment below to use CPU instead of GPU
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# import tensorflow as tf

# config = tf.ConfigProto(intra_op_parallelism_threads=4,
#                         inter_op_parallelism_threads=4, 
#                         allow_soft_placement=True,
#                         device_count = {'CPU' : 1,
#                                         'GPU' : 0}
#                        )

from keras.layers import GRU, LSTM, CuDNNLSTM
from price_pred import PricePrediction

ticker = "USD"

p = PricePrediction("USD", epochs=100, cell=LSTM, optimizer="adam", n_layers=3, 
                    units=256, loss="mae")

# train model
# add path for the train datasets
p.train('../data/output_data.csv')
# predict model
p.predict()

# print("Mean Absolute Error:", p.get_MAE())
# print("Mean Squared Error:", p.get_MSE())
# p.plot_test_set()