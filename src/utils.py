import os
import argparse
import time
import datetime
import numpy as np
import pandas as pd
import preprocessing as pp


# 检查model
def model_check(X,y_map,n,df,model):
    for i in range(n):
        chk=np.random.randint(df.shape[0])
        assert df.metadata_similar_artists.iloc[chk][0] == y_map[np.argmax(
            model.predict(X[chk].reshape(1,-1)))]


def save_lookup_file(df):
    lookupDF=df[
        ['metadata_songs_song_id','metadata_songs_artist_id','metadata_songs_title','metadata_songs_artist_name',
         'musicbrainz_songs_year','metadata_songs_release']]

    pp.convert_byte_data(lookupDF)

    lookupDF.to_hdf('../frontend/data/lookup.h5',key='df',mode='w')

    pd.read_hdf('../frontend/data/lookup.h5','df')


def setup_model_dir():
    t=time.time()
    dt=datetime.datetime.fromtimestamp(t).strftime('%Y%m%d%H%M%S')
    path='../model/'+dt
    os.mkdir(path)
    os.mkdir(path+'/preprocessing')

    return path


def arg_parser():
    # 使用description和-h方法分析设置参数
    parser=argparse.ArgumentParser(
        description='Music recommendation System')
    # 添加文件数量int变量
    parser.add_argument('-s','--size',default=10000,type=int,nargs='?',
                        help='the number of files to use for training')
    # 添加初始化变量
    parser.add_argument('-i','--initialize',default=False,type=bool,nargs='?',
                        help='flag to run initial setup for web app')
    # 返回args
    args=parser.parse_args()
    return args
