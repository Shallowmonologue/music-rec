#!/usr/bin/env python3
# coding: utf-8
"""
main.py
03-29-19
jack skrable
"""

# Library imports
import time
import h5py
import pandas as pd
import numpy as np

# Custom imports
import utils
import plot
import read_h5 as read
import preprocessing as pp
import neural_net as nn
import kmeans as km

args = utils.arg_parser()

# 读取MSD数据集的h5文件，并统计所用时间
t_start = time.time()
df = read.h5_to_df('../data/MillionSongSubset/data', args.size, args.initialize)
t_extract = time.time()
print('\nGot', len(df.index), 'songs in', round((t_extract-t_start), 2), 'seconds.')

# 设定预处理文件与训练模型的导出目录/model
path = utils.setup_model_dir()

# 将数据变换为可供神经网络训练的向量
print('Pre-processing extracted song data...')
df = pp.convert_byte_data(df)
df = pp.create_target_classes(df)

# 训练集shuffle
for i in range(5):
    df = df.iloc[np.random.permutation(len(df))]
df = df.fillna(0)

# 将数据调整为Numpy矩阵，按列归一化
X, y, y_map = pp.vectorize(df, 'target', path)
t_preproc = time.time()
print('Cleaned and processed', len(df.index), 'rows in',
      round((t_preproc - t_extract), 2), 'seconds.')

# 训练模型
print('Training model...')
print('[', X.shape[1], '] x [', np.unique(y).size, ']')
model_simple = nn.deep_nn(pp.scaler(X, 'robust', path), y, 'std', path)
t_nn = time.time()
print('Model trained costs', round((t_nn - t_preproc), 2), 'seconds.')

# 验证模型
print('Evaluating model and saving class probabilities...')
predDF = pd.DataFrame.from_records(model_simple.predict(pp.scaler(X, 'robust')))
predDF.to_pickle(path + '/model_prob.pkl')

# 进行k均值聚类，并通过神经网络发送分类数据
###############################################################################
clusters = 18
print('Applying k-Means classifier with', clusters, 'clusters...')
kmX = km.kmeans(pp.scaler(X, 'robust', path), clusters)
print('Complete.')
print('Training neural network...')
print('[', kmX.shape[1], '] x [', np.unique(y).size, ']')
model_classified = nn.deep_nn(kmX, y, 'hyb', path)
t_km = time.time()
print('Hybrid k-Means neural network trained in',
      round((t_km - t_nn), 2), 'seconds.')


# 打印测试结果
###############################################################################
# plot(X, kmX[:,-1])
plot.plot_nn_training(path, 'loss')
plot.plot_nn_training(path, 'accuracy')