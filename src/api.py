import os
import json
import tensorflow as tf
import numpy as np
import pandas as pd
import flask
import joblib
import neural_net as nn
import read_h5 as read
import preprocessing as pp

# 初始化flask
app = flask.Flask(__name__)
model = None
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def load_model():
    global lookupDF
    global song_file_map
    global column_maps
    global max_list
    global model
    global scaler
    global probDF

    # 载入模型
    model = nn.load_model('../model/working/std')

    # 载入预处理文件
    with open('../data/song-file-map.json', 'r') as f:
        song_file_map = json.load(f)
    with open('../model/working/preprocessing/maps.json', 'r') as f:
        column_maps = json.load(f)
    with open('../model/working/preprocessing/max_list.json', 'r') as f:
        max_list = json.load(f)

    scaler = joblib.load('../model/working/preprocessing/robust.scaler')

    # 载入歌曲ID对应表
    lookupDF = pd.read_hdf('../frontend/data/lookup.h5', 'df')

    # 模型预测
    probDF = pd.read_pickle('..//model/working/model_prob.pkl')


def process_metadata_list(col):
    x_map = column_maps[col.name]
    max_len = max_list[col.name]
    col = col.apply(lambda x: pp.lookup_discrete_id(x, x_map))
    col = col.apply(lambda x: np.pad(x, (0, max_len - x.shape[0]), 'constant'))
    xx = np.stack(col.values)
    return xx


def preprocess_predictions(df):
    print('Vectorizing dataframe...')
    for col in df:
        if df[col].dtype == 'O':
            if type(df[col].iloc[0]) is str:
                xx = pp.lookup_discrete_id(df[col], column_maps[col])
                xx = xx.reshape(-1, 1)
            elif col.split('_')[0] == 'metadata':
                xx = process_metadata_list(df[col])
            else:
                xx = pp.process_audio(df[col])

        else:
            xx = df[col].values[..., None]

        xx = xx / (np.linalg.norm(xx) + 0.00000000000001)
        try:
            output = np.hstack((output, xx))
        except NameError:
            output = xx

    return output


def get_recs(song_ids):

    song_ids = song_ids.split(',')
    # 根据歌曲ID查询文件名
    files = [song_file_map[id] for id in song_ids]

    # 提取文件数据
    df = read.extract_song_data(files)
    df = pp.convert_byte_data(df)
    df = df.fillna(0)

    # 向量化并进行预测
    X = preprocess_predictions(df)
    print('Model predicting...')
    predictions = model.predict(X)

    classes = [column_maps['target'][i.argmax()] for i in predictions]

    # model_prob = probDF[probDF.columns[:-1]].values
    model_prob = probDF[probDF.columns].values

    rec_ids = [probDF.iloc[np.argmin(np.min(np.sqrt((model_prob-pred)**2),axis=1))].name
                for pred in predictions]

    # recs = lookupDF.loc[lookupDF.metadata_songs_song_id.isin(rec_ids)].to_dict('records')
    recs = lookupDF.loc[rec_ids].to_dict('records')

    return classes, recs


@app.route("/recommend", methods=["GET"])
def recommend():
    # 初始化响应
    data = {"success": False}

    # 若有GET请求
    if flask.request.method == "GET":

        # 搜索歌曲ID
        song_ids = flask.request.args.get('songs')
        # 获取分类和推荐结果
        classes, recs = get_recs(song_ids)

    print(classes)
    print(recs)

    # 创建响应实体
    data['entity'] = {'classes': classes}
    data['entity'].update({'recommendations': recs})

    # 请求成功
    data["success"] = True

    # JSONify
    response = flask.jsonify(data)
    # Allow CORS
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response


@app.route("/lookup", methods=["GET"])
def lookup():
    # 初始化
    data = {"success": False}

    # 确保图片已被正确上传
    if flask.request.method == "GET":

        # 获取记录数据
        data['entity'] = lookupDF.to_dict('records')

        # 请求成功
        data["success"] = True

    # JSONify
    response = flask.jsonify(data)
    # CORS
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response


# 加载模型并启动服务器
if __name__ == "__main__":
    print(" * Starting Flask server and loading Keras model...")
    print(" * Please wait until server has fully started")

    # 载入模型与相关文件
    load_model()

    print(' * Server is active')
    # 启动
    app.run(host='0.0.0.0', port=5001)
