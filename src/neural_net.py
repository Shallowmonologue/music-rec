import os
import time
import datetime
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Flatten,LeakyReLU,BatchNormalization,Activation,Softmax
from tensorflow.keras.callbacks import TensorBoard,CSVLogger
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from tensorflow.keras.applications import imagenet_utils


# 设定优化函数
def set_opt(OPT, lr):

    if OPT == 'sgd':
        opt = optimizers.SGD(lr=lr, decay=1e-6, momentum=0.9,
                             nesterov=True)
    elif OPT == 'adam':
        opt = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999,
                              epsilon=None, decay=1e-6, amsgrad=False)
    elif OPT == 'adamax':
        opt = optimizers.Adamax(lr=lr, beta_1=0.9, beta_2=0.999,
                                epsilon=None, decay=0.0)
    elif OPT == 'adagrad':
        opt = optimizers.Adagrad(lr=lr, epsilon=None, decay=0.0)
    elif OPT == 'adadelta':
        opt = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)

    return opt


# 模型的构建与训练
def deep_nn(X, y, label, path=None):

    K.clear_session()

    try:
        path += ('/' + label)
        os.mkdir(path)
    except FileExistsError:
        print(path, 'already exists.')

    # 定义超参数
    lr = 0.0001
    epochs = 500
    batch_size = 50
    OPT = 'adadelta'

    # 设定模型储存的文件夹名
    t = time.time()
    dt = datetime.datetime.fromtimestamp(t).strftime('%Y%m%d%H%M%S')
    name = '_'.join([OPT, str(epochs), str(batch_size), dt])

    # 计算分类权重以提高准确率
    class_weights = dict(enumerate(compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)))
    swm = np.array([class_weights[i] for i in y])

    # 将结果向量转化为矩阵形式
    y = to_categorical(y, num_classes=len(class_weights))

    # 初始化输入矩阵大小
    in_size = X.shape[1]

    # 3层隐含层的变量设置
    hidden_1_size = 400
    hidden_2_size = 150
    hidden_3_size = 50

    # Modify this when increasing artist list target
    out_size = y.shape[1]

    # 分割训练集、验证集、测试集
    print('Splitting to train, test, and validation sets...')
    X_train, X_test, X_valid = np.split(X, [int(.6 * len(X)), int(.8 * len(X))])
    y_train, y_test, y_valid = np.split(y, [int(.6 * len(y)), int(.8 * len(y))])

    # 初始化构造函数
    model = Sequential()

    # 添加输入层
    model.add(Dense(in_size, input_shape=(in_size,), activation='relu',
                    kernel_initializer='normal', kernel_constraint=max_norm(3)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # (调用)添加隐含层
    def add_hidden_layer(s, b=False, a=0.3, d=0.0):
        if d > 0:
            model.add(Dense(s, kernel_initializer='normal', kernel_constraint=max_norm(3)))
        else:
            model.add(Dense(s))
        if b:
            model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=a))
        model.add(Dropout(d))

    # 添加隐含层
    add_hidden_layer(in_size, True, 0.3, 0.5)
    add_hidden_layer(hidden_2_size, True, 0.3, 0.3)
    add_hidden_layer(hidden_2_size, True, 0.3, 0.3)
    add_hidden_layer(hidden_3_size, True, 0.3, 0.1)

    # 添加输出层
    model.add(Dense(out_size))
    model.add(BatchNormalization())
    model.add(Softmax(axis=-1))

    # 设定优化器
    opt = set_opt(OPT, lr)

    # loss等其他模型相关设定
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'],
                  sample_weight_mode=swm)

    # (调参用)
    tensorboard = TensorBoard(log_dir=str('../logs/'+label+'/'+name+'.json'),
                              histogram_freq=1,
                              write_graph=True,
                              write_images=False)

    # 储存model训练的日志位置
    csv_logs = CSVLogger(filename=path+'/logs.csv',
                         separator=',',
                         append=False)

    # 开始进行训练
    print('Training...')
    model.fit(X_train, y_train, validation_data=(X_valid, y_valid),
              epochs=epochs, batch_size=batch_size, verbose=1,
              shuffle=True, callbacks=[csv_logs])

    # 进行验证
    print('Evaluating...')
    y_pred = model.predict(X_test)
    score = model.evaluate(X_test, y_test,verbose=1)
    print(score)

    # 保存模型json文件
    print('Saving model...')
    model_json = model.to_json()
    with open(path + '/model.json', 'w') as file:
        file.write(model_json)

    # 通过h5格式保存模型权重，同时保存为csv文件
    model.save_weights(path + '/weights.h5')
    np.savetxt(path + '/sample_weights.csv', swm, delimiter=',')

    # 保存超参数
    with open(path + '/hyperparams.csv', 'w') as file:
        file.write(','.join([str(lr), OPT]))
    print('Model saved to disk')

    return model


# 载入模型
def load_model(path):

    # 获取神经网络结构
    with open(path + '/model.json','r') as file:
        structure = file.read()
    model = model_from_json(structure)

    # 获取网络权重
    model.load_weights(path + '/weights.h5')

    # 获取分类权重
    swm = np.genfromtxt(path + '/sample_weights.csv', delimiter=',')

    # 获取超参数
    with open(path + '/hyperparams.csv', 'r') as file:
        lr, OPT = file.read().split(',')

    # 构建优化器
    opt = set_opt(OPT, float(lr))

    # 模型设定
    model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'],
              sample_weight_mode=swm)

    return model





