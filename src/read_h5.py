import json
import os
import sys
import tables
import glob
import pandas as pd
import numpy as np


# 可视化的进度条
def progress(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = '#' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write('[%s] %s%s %s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush()


# 读取目录下的所有h5文件
def get_all_files(basedir, ext='.h5'):
    print('Getting list of all h5 files in',basedir)
    allfiles = []
    for root, dirs, files in os.walk(basedir):
        files = glob.glob(os.path.join(root, '*'+ext))
        for f in files:
            allfiles.append(os.path.abspath(f))
    return allfiles


# 解压h5文件,并构建dataframe
def extract_song_data(files):

    # 初始化
    df = pd.DataFrame()
    size = len(files)
    print(size, 'files found.')

    # 循环遍历文件
    for i, f in enumerate(files):
        # 更新可视化进度条
        progress(i, size, 'of files processed')
        # 储存文件
        s_hdf = pd.HDFStore(f)
        # 单独的dataframe
        data = pd.DataFrame()
        for item in s_hdf.root._f_walknodes():
            # 每列的列名
            name = item._v_pathname[1:].replace('/','_')
            # 储存数组
            if type(item) is tables.earray.EArray:
                data[name] = [np.array(item)]
            # 储存表格
            elif type(item) is tables.table.Table:
                # 获取所有列
                cols = item.coldescrs.keys()
                for row in item:
                    for col in cols:
                        col_name = '_'.join([name,col])
                        try:
                            data[col_name] = row[col]
                        except Exception as e:
                            print(e)

        # 添加到主dataframe中
        df = df.append(data, ignore_index=True)
        # 关闭store
        s_hdf.close()

    # 去除部分不需要的数据
    # df = df[['metadata_songs_artist_id','metadata_songs_title',
    # 'musicbrainz_songs_year','metadata_artist_terms',
    # 'analysis_songs_analysis_sample_rate','metadata_songs_artist_location',
    # 'analysis_sections_confidence','analysis_sections_start','analysis_segments_start',
    # 'analysis_segments_timbre','analysis_segments_pitches','analysis_songs_tempo',
    # \'analysis_bars_confidence','analysis_bars_start','analysis_beats_confidence',
    # 'analysis_beats_start','analysis_songs_duration','analysis_songs_energy',
    # 'analysis_songs_key','analysis_songs_key_confidence','analysis_songs_time_signature',
    # 'analysis_songs_time_signature_confidence','metadata_similar_artists']]
    df.drop(['musicbrainz_artist_mbtags_count','musicbrainz_artist_mbtags',
             'musicbrainz_songs_idx_artist_mbtags'], inplace=True, axis=1)

    return df


# 网页读取歌单
def get_song_file_map(files):

    songmap = {}
    size = len(files)
    print(size, 'files found.')

    # 遍历歌单
    for i, f in enumerate(files):
        # 更新可视化进度条
        progress(i, size, 'of files processed')
        # 储存文件store
        s_hdf = pd.HDFStore(f)
        song_id = s_hdf.root.metadata.songs[0]['song_id'].astype('U')
        filepath = s_hdf.filename
        songmap.update({song_id: s_hdf.filename})
        # 关闭文件store
        s_hdf.close()

    with open('../data/song-file-map.json', 'w') as file:
        json.dump(songmap, file, sort_keys=True, indent=2)

    return songmap

# 三元组数据集
def get_user_taste_data(filename):
    tasteDF = pd.read_csv('../TasteProfile/train_triplets_SAMPLE.txt', sep='\t', header=None, names={'user,song,count'})

    return tasteDF


# 读取h5文件并将其转化为dataframe
def h5_to_df(basedir, limit=None, init=False):
    files = get_all_files(basedir, '.h5')
    files = files if limit is None else files[:limit]
    df = extract_song_data(files)

    if init:
        get_song_file_map(files)

    return df