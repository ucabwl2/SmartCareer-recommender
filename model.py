import pandas as pd
import random
import numpy as np
import paddle
from paddle.nn import Linear, Embedding, Conv2D
import paddle.nn.functional as F
import math
from PIL import Image
import pickle
import shutil
import os
from job_data import JobDataset
from Constants import *
from SimCSE import SimcseModel
from tqdm import tqdm
import json 
import pandas as pd
data_root = Data_path

def get_bert_embedding(data_path,sheet_name,row_to_keep_list,table_key,column_to_embedding,out_path):
    data = pd.read_excel(data_path,engine="openpyxl",sheet_name=sheet_name)
    data_keep = data[[table_key,column_to_embedding]]
    data_in_rating = data_keep[data_keep[table_key].isin(row_to_keep_list)]
    print(f"Now we have columns {data_keep.columns}, and {len(data_in_rating)} records")
    data_in_rating_dic = data_in_rating[[table_key,column_to_embedding]].to_dict('records')
    model = SimcseModel()
    for dt in tqdm(data_in_rating_dic):
        if isinstance(dt[column_to_embedding],str):
            dt[column_to_embedding + '_embedding'] = model(dt[column_to_embedding]).tolist()
    with open(out_path, 'w') as fout:
        json.dump(data_in_rating_dic, fout)

class Model(paddle.nn.Layer):
    def __init__(self,use_poster, use_usr_title, use_usr_state, use_job_city, use_job_exp_year,fc_sizes):
        super(Model, self).__init__()
        
        # 将传入的name信息和bool型参数添加到模型类中
        self.use_mov_poster = use_poster
        self.use_usr_title = use_usr_title
        self.use_usr_state = use_usr_state
        self.use_job_city = use_job_city
        self.use_job_exp_year = use_job_exp_year
        self.fc_sizes = fc_sizes
        
        #Embedding_path
        if not os.path.exists(data_root+"all_job_ProjectDescription_embedding.json"):
            jobs = pd.read_excel(data_root+"jobs.xlsx",engine="openpyxl",sheet_name='summary')
            job_list = set(jobs['jobid'])
            print("generating job description embedding")
            get_bert_embedding(data_root+"jobs.xlsx",'summary',list(job_list),'jobid','ProjectDescription',data_root+'all_job_ProjectDescription_embedding.json')
        
        self.job_proj_emb = pd.read_json(data_root+"all_job_ProjectDescription_embedding.json")
        avg_emb = [np.array([i[0] for i in self.job_proj_emb['ProjectDescription_embedding'].dropna().values]).mean(axis=0).tolist()]
        self.job_proj_emb['ProjectDescription_embedding'] = self.job_proj_emb['ProjectDescription_embedding'].apply(lambda d: d if isinstance(d, list) else avg_emb)

        if not os.path.exists(data_root+"usr_sum_emb.json"):
            usrs = pd.read_excel(data_root+"freelancers.xlsx",engine="openpyxl",sheet_name='summary')
            usr_list = set(usrs['FreelancerID'])
            print("generating user summary embedding")
            get_bert_embedding(data_root+"freelancers.xlsx",'freelancer summary',list(job_list),'FreelancerID','Summary',data_root+'usr_sum_emb.json')
        self.usr_summary_emb = pd.read_json(data_root+"usr_sum_emb.json")
        # 获取数据集的信息，并构建训练和验证集的数据迭代器
        Dataset = JobDataset(self.use_mov_poster)
        self.Dataset = Dataset
        self.trainset = self.Dataset.train_dataset
        self.valset = self.Dataset.valid_dataset
        self.train_loader = self.Dataset.load_data(dataset=self.trainset, mode='train')
        self.valid_loader = self.Dataset.load_data(dataset=self.valset, mode='valid')

        """ define network layer for embedding usr info """
        USR_ID_NUM = Dataset.max_usr_id + 1
        # 对用户ID做映射，并紧接着一个Linear层
        self.usr_emb = Embedding(num_embeddings=USR_ID_NUM, embedding_dim=32, sparse=False)
        self.usr_fc = Linear(in_features=32, out_features=32)
        self.usr_summary_fc = Linear(in_features=768, out_features=768)
        
        # 对usr title 信息做映射，并紧接着一个Linear层
        USR_TITLE_DICT_SIZE = len(Dataset.usr_title_info) + 1
        self.usr_title_emb = Embedding(num_embeddings=USR_TITLE_DICT_SIZE, embedding_dim=32, sparse=False)
        self.usr_title_conv = Conv2D(in_channels=1, out_channels=1, kernel_size=(3, 1), stride=(2,1), padding=0)
        self.usr_title_conv2 = Conv2D(in_channels=1, out_channels=1, kernel_size=(3, 1), stride=1, padding=0)
        
        # 对usr state信息做映射，并紧接着一个Linear层
        USR_STATE_DICT_SIZE = Dataset.max_usr_state + 1
        self.usr_state_emb = Embedding(num_embeddings=USR_STATE_DICT_SIZE, embedding_dim=16)
        self.usr_state_fc = Linear(in_features=16, out_features=16)
        
        # # 对用户职业信息做映射，并紧接着一个Linear层
        # USR_JOB_DICT_SIZE = Dataset.max_usr_job + 1
        # self.usr_job_emb = Embedding(num_embeddings=USR_JOB_DICT_SIZE, embedding_dim=16)
        # self.usr_job_fc = Linear(in_features=16, out_features=16)
        
        # 新建一个Linear层，用于整合用户数据信息
        self.usr_combined = Linear(in_features=848, out_features=200)
        
        """ define network layer for embedding job info """
        # 对电影ID信息做映射，并紧接着一个Linear层
        JOB_DICT_SIZE = Dataset.max_job_id + 1
        self.job_emb = Embedding(num_embeddings=JOB_DICT_SIZE, embedding_dim=32)
        self.job_fc = Linear(in_features=32, out_features=32)
        self.job_proj_fc = Linear(in_features=768, out_features=768)
        
        # 对电影类别做映射
        JOB_CITY_DICT_SIZE = Dataset.max_job_city + 1
        self.job_city_emb = Embedding(num_embeddings=JOB_CITY_DICT_SIZE, embedding_dim=16, sparse=False)
        self.job_city_fc = Linear(in_features=16, out_features=16)
        
        # 对电影名称做映射
        JOB_EXP_YEAR_DICT_SIZE = 9 + 1
        self.job_exp_year_emb = Embedding(num_embeddings=JOB_EXP_YEAR_DICT_SIZE, embedding_dim=16, sparse=False)
        self.job_exp_year_fc = Linear(in_features=16, out_features=16)
        
        # 新建一个FC层，用于整合电影特征
        self.job_concat_embed = Linear(in_features=832, out_features=200)

        user_sizes = [200] + self.fc_sizes
        acts = ["relu" for _ in range(len(self.fc_sizes))]
        self._user_layers = []
        for i in range(len(self.fc_sizes)):
            linear = paddle.nn.Linear(
                in_features=user_sizes[i],
                out_features=user_sizes[i + 1],
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Normal(
                        std=1.0 / math.sqrt(user_sizes[i]))))
            self._user_layers.append(linear)
            if acts[i] == 'relu':
                act = paddle.nn.ReLU()
                self._user_layers.append(act)
        
        #电影特征和用户特征使用了不同的全连接层，不共享参数
        job_sizes = [200] + self.fc_sizes
        acts = ["relu" for _ in range(len(self.fc_sizes))]
        self._job_layers = []
        for i in range(len(self.fc_sizes)):
            linear = paddle.nn.Linear(
                in_features=job_sizes[i],
                out_features=job_sizes[i + 1],
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Normal(
                        std=1.0 / math.sqrt(job_sizes[i]))))
            self._job_layers.append(linear)
            if acts[i] == 'relu':
                act = paddle.nn.ReLU()
                self._job_layers.append(act)
        
    # 定义计算用户特征的前向运算过程
    def get_usr_feat(self, usr_var):
        """ get usr features"""
        # 获取到用户数据
        usr_id, usr_title, usr_state = usr_var
        # 将用户的ID数据经过embedding和Linear计算，得到的特征保存在feats_collect中
        feats_collect = []
        batch_size = usr_id.shape[0]
        #usr_summary_embeeding
        tempid = [int(i) for i in usr_id]
        usr_summary = [self.usr_summary_emb[self.usr_summary_emb['FreelancerID']==i]['Summary_embedding'].values[0] for i in tempid]
        # usr_summary = self.usr_summary_emb['Summary_embedding'].loc[self.usr_summary_emb['FreelancerID'].isin(tempid)]
        # usr_summary = usr_summary.to_list()
        # usr_summary = [self.usr_summary_emb.loc[self.usr_summary_emb['FreelancerID']==i, ['FreelancerID']] for i in tempid]
        # print(usr_summary)
        usr_summary = paddle.to_tensor(usr_summary)
        
        
        
        usr_summary = self.usr_summary_fc(usr_summary)
        usr_summary = F.relu(usr_summary)
        feats_collect.append(usr_summary)
    
        #usr_id embedding
        usr_id = self.usr_emb(usr_id)
        usr_id = self.usr_fc(usr_id)
        usr_id = F.relu(usr_id)
        feats_collect.append(usr_id)
    

        
        if self.use_usr_title:
            # 计算电影名字的特征映射，对特征映射使用卷积计算最终的特征
            usr_title = self.usr_title_emb(usr_title)
            usr_title = F.relu(self.usr_title_conv2(F.relu(self.usr_title_conv(usr_title))))
            usr_title = paddle.sum(usr_title, axis=2, keepdim=False)
            usr_title = F.relu(usr_title)
            usr_title = paddle.reshape(usr_title, [batch_size, -1])
            feats_collect.append(usr_title)
        
        
        # 选择是否使用用户的年龄-职业特征
        if self.use_usr_state:
            # 计算用户的年龄特征，并保存在feats_collect中
            usr_state = self.usr_state_emb(usr_state)
            usr_state = self.usr_state_fc(usr_state)
            usr_state = F.relu(usr_state)
            feats_collect.append(usr_state)
        
        # 将用户的特征级联，并通过Linear层得到最终的用户特征
        # try:
        if True:
            usr_feat = paddle.concat(feats_collect, axis=1)
        # except:
        #     for feat in feats_collect:
        #         print(feat.shape)
        #     usr_feat = paddle.concat(feats_collect, axis=1)

        user_features = F.tanh(self.usr_combined(usr_feat))
        #通过3层全连接层，获得用于计算相似度的用户特征
        for n_layer in self._user_layers:
            user_features = n_layer(user_features)

        return user_features

        # 定义电影特征的前向计算过程
    def get_job_feat(self, job_var):
        """ get movie features"""
        # 获得电影数据
        
        job_id, job_city, job_exp_year= job_var
        feats_collect = []
        # 获得batchsize的大小
        batch_size = job_id.shape[0]
        
        #prob_description_embeeding
        tempid = [int(i) for i in job_id]
        # print(tempid)
        # proj_emb = self.job_proj_emb['ProjectDescription_embedding'].loc[self.job_proj_emb['jobid'].isin(tempid)]
        # print('====asda',proj_emb)# batch_size,768
        # proj_emb = proj_emb.apply(lambda x:x[0])
        # proj_emb = proj_emb.to_list()
        # proj_emb = paddle.to_tensor(proj_emb)
        proj_emb = [self.job_proj_emb[self.job_proj_emb['jobid']==i]['ProjectDescription_embedding'].values[0][0] for i in tempid]
        # proj_emb = [self.job_proj_emb.loc[self.job_proj_emb['jobid']==i,['ProjectDescription_embedding']][0] for i in tempid]
        proj_emb = paddle.to_tensor(proj_emb)
        proj_emb = self.job_proj_fc(proj_emb)
        proj_emb = F.relu(proj_emb)
        feats_collect.append(proj_emb)
        tempid = []
        
        # 计算电影ID的特征，并存在feats_collect中
        job_id = self.job_emb(job_id)
        job_id = self.job_fc(job_id)
        job_id = F.relu(job_id)
        feats_collect.append(job_id)
        
        # 如果使用电影的种类数据，计算电影种类特征的映射
        if self.use_job_city:
            # 计算电影种类的特征映射，对多个种类的特征求和得到最终特征
            job_city = self.job_city_emb(job_city)
            job_city = self.job_city_fc(job_city)
            job_city = F.relu(job_city)
            feats_collect.append(job_city)

        if self.use_job_exp_year:
            # 计算电影名字的特征映射，对特征映射使用卷积计算最终的特征
            job_exp_year = self.job_exp_year_emb(job_exp_year)
            job_exp_year = self.job_exp_year_fc(job_exp_year)
            job_exp_year = F.relu(job_exp_year)
            feats_collect.append(job_exp_year)
            
        # 使用一个全连接层，整合所有电影特征，映射为一个200维的特征向量
        job_feat = paddle.concat(feats_collect, axis=1)
        job_features = F.tanh(self.job_concat_embed(job_feat))
        #通过3层全连接层，获得用于计算相似度的电影特征
        for n_layer in self._job_layers:
            job_features = n_layer(job_features)

        return job_features
    
    # 定义个性化推荐算法的前向计算
    def forward(self, usr_var, job_var):
        # 计算用户特征和电影特征
        user_features = self.get_usr_feat(usr_var)
        job_features = self.get_job_feat(job_var)
        # print(user_features.shape, job_features.shape)

        # 根据计算的特征计算相似度
        res = F.common.cosine_similarity(user_features, job_features).reshape([-1,1])
        # 将相似度扩大范围到和电影评分相同数据范围
        # res = paddle.scale(res, scale=5)
        return user_features, job_features, res