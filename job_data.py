import pandas as pd
import random
import numpy as np
import paddle
from paddle.nn import Linear, Embedding, Conv2D
import paddle.nn.functional as F
import math
from PIL import Image
import pickle
from Constants import *
import os
data_root = Data_path

class JobDataset(object):
    def __init__(self, use_poster):
        self.use_poster = use_poster
        usr_info_path = data_root+"freelancers.xlsx"
        if use_poster:
            rating_path = data_root+"new_rating.txt"
        else:
            rating_path = data_root+"post_rating.xlsx"
            if not os.path.exists(rating_path):
                from sample_rating import result
        job_info_path = data_root+"jobs.xlsx"
        for required_file in [usr_info_path, rating_path, job_info_path]:
            assert os.path.exists(required_file)
        
        self.job_info, self.job_city = self.get_job_info(job_info_path)
        # 记录电影的最大ID
        self.max_job_id = np.max(list(map(int, self.job_info.keys())))
        self.max_job_city = np.max(list(map(int, self.job_city.values())))
        # 记录用户数据的最大ID
        self.max_usr_id = 0
        # 得到用户数据
        self.usr_info, self.usr_title_info, self.usr_state_info = self.get_usr_info(usr_info_path)
        self.max_usr_title = np.max([self.usr_title_info[k] for k in self.usr_title_info])
        self.max_usr_state = np.max(list(map(int, self.usr_state_info.values())))
        # 得到评分数据
        self.rating_info = self.get_rating_info(rating_path)
        # 构建数据集 
        self.dataset = self.get_dataset(usr_info=self.usr_info,
                                        job_info=self.job_info,
                                        rating_info=self.rating_info
                                       )
        # 划分数据集，获得数据加载器
        self.train_dataset = self.dataset[:int(len(self.dataset)*0.9)]
        self.valid_dataset = self.dataset[int(len(self.dataset)*0.9):]
        print("##Total dataset instances: ", len(self.dataset))
        print("##MovieLens dataset information: \nusr num: {}\n"
              "job num: {}".format(len(self.usr_info),len(self.job_info)))
    # 得到电影数据
    def get_job_info(self, path):
        # 打开文件，编码方式选择ISO-8859-1，读取所有数据到data中 
        df_job = pd.read_excel(open(path, 'rb'),
                             sheet_name='summary')
        # 建立三个字典，分别用户存放电影所有信息，电影的名字信息、类别信息
        job_info, job_city = {}, {}
        df_job['Value'].replace(['Min of 1 year','Min of 3 years','Min of 5 years','Min of 7 years','Min of 9 years'],[1,3,5,7,9],inplace= True)
        # 对电影名字、类别中不同的单词计数
        count_c = 1
        
        for i in df_job.index:
            job_id = df_job['jobid'][i]
            
            if df_job['WorkLocationCity'][i] not in job_city:
                job_city[df_job['WorkLocationCity'][i]] = count_c
                count_c +=1
            job_info[job_id] = {'job_id': int(job_id),
                                'job_city': job_city[df_job['WorkLocationCity'][i]],
                                'job_exp_year': df_job['Value'][i]
                                }

        return job_info, job_city

    def get_usr_info(self, path):
        # 打开文件，读取所有行到data中
        data = pd.read_excel(open(path, 'rb'),
                             sheet_name='freelancer summary') 
        # 建立用户信息的字典
        state_info, title_info, usr_info = {}, {}, {}
        max_state_len, max_title_len= 0,0
        count_s, count_t = 1,1
        data['State'] = data['State'].fillna('')
        data['Title'] = data['Title'].mask(pd.to_numeric(data['Title'], errors='coerce').notna()).fillna('')
        data['Title'].replace(['/','& ','|','- ',', ','Sr.'],[' ',' ',' ',' ',' ','Senior'], inplace = True)
        
        for i in data.index:
            usr_id = data['FreelancerID'][i]
            each_title = data['Title'][i].split()
            # print(each_title)
            max_title_len = max(max_title_len, len(each_title))

            if data['State'][i] not in state_info:
                state_info[data['State'][i]] = count_s
                count_s += 1

            for t in each_title:
                if t not in title_info:
                    title_info[t] = count_t
                    count_t +=1

            v_title = [title_info[k] for k in each_title]
            # print(usr_id,v_title)
            while len(v_title)<39:
                v_title.append(0)
            try:
                usr_info[usr_id] = {'usr_id': int(usr_id),
                                    'title': v_title,
                                    'state': state_info[data['State'][i]]
                                    }
                                
                self.max_usr_id = max(self.max_usr_id, int(usr_id))
            except:
                continue
        return usr_info, title_info, state_info
    # 得到评分数据
    def get_rating_info(self, path):
        # 打开文件，读取所有行到data中
        df_rating = pd.read_excel(open(path, 'rb'))
        # 创建一个字典
        rating_info = {}
        for i in df_rating.index:
            usr_id, jobid, rating = df_rating['FreelancerID'][i], df_rating['jobid'][i],df_rating['rating'][i]
            if usr_id in self.usr_info.keys():
                if usr_id not in rating_info.keys():
                    rating_info[usr_id] = {jobid:float(rating)}
                else:
                    rating_info[usr_id][jobid] = float(rating)
        return rating_info
    # 构建数据集
    def get_dataset(self, usr_info, job_info, rating_info):
        trainset = []
        # 按照评分数据的key值索引数据
        for usr_id in rating_info.keys():
            usr_ratings = rating_info[usr_id]
            for jobid in usr_ratings:
                trainset.append({'usr_info': usr_info[usr_id],
                                 'job_info': job_info[jobid],
                                 'scores': usr_ratings[jobid]})
        return trainset

    def load_data(self, dataset=None, mode='train', BATCHSIZE=256):
        use_poster = False

        data_length = len(dataset)
        index_list = list(range(data_length))
        # 定义数据迭代加载器
        def data_generator():
            # 训练模式下，打乱训练数据
            if mode == 'train':
                random.shuffle(index_list)
            # 声明每个特征的列表
            usr_id_list,usr_title_list,usr_state_list,usr_job_list = [], [], [], []
            job_id_list,job_city_list,job_exp_year_list = [], [], []
            score_list = []
            # 索引遍历输入数据集
            for idx, i in enumerate(index_list):
                # 获得特征数据保存到对应特征列表中
                usr_id_list.append(dataset[i]['usr_info']['usr_id'])
                usr_title_list.append(dataset[i]['usr_info']['title'])
                usr_state_list.append(dataset[i]['usr_info']['state'])
                # usr_job_list.append(dataset[i]['usr_info']['job'])

                job_id_list.append(dataset[i]['job_info']['job_id'])
                job_city_list.append(dataset[i]['job_info']['job_city'])
                job_exp_year_list.append(dataset[i]['job_info']['job_exp_year'])
                job_id = dataset[i]['job_info']['job_id']



                score_list.append(int(dataset[i]['scores']))
                # 如果读取的数据量达到当前的batch大小，就返回当前批次
                
                if len(usr_id_list)==BATCHSIZE:
                    # 转换列表数据为数组形式，reshape到固定形状
                    usr_id_arr = np.array(usr_id_list)
                    usr_title_arr = np.reshape(np.array(usr_title_list), [BATCHSIZE, 1, 39]).astype(np.int64)
                    usr_state_arr = np.array(usr_state_list)
                    # usr_job_arr = np.array(usr_job_list)

                    job_id_arr = np.array(job_id_list)
                    job_city_arr = np.array(job_city_list)
                    job_exp_year_arr = np.array(job_exp_year_list)

                    scores_arr = np.reshape(np.array(score_list), [-1, 1]).astype(np.float32)

                    # 返回当前批次数据
                    yield [usr_id_arr, usr_title_arr, usr_state_arr], \
                           [job_id_arr, job_city_arr, job_exp_year_arr], scores_arr

                    # 清空数据
                    usr_id_list, usr_title_list, usr_state_list = [], [], []
                    job_id_list,job_city_list,job_exp_year_list = [], [], []
                    score_list = []
                    mov_poster_list = []
        return data_generator