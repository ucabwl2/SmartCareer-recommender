from PIL import Image
# 加载第三方库Pickle，用来保存Python数据到本地
import pickle
import paddle
import numpy as np
from model import Model
from job_data import JobDataset
# 定义特征保存函数
def get_usr_job_features(model, params_file_path):
    model_state_dict = paddle.load(params_file_path)
    model.load_dict(model_state_dict)
    paddle.set_device('cpu') 
    usr_pkl = {}
    job_pkl = {}
    
    # 定义将list中每个元素转成tensor的函数
    def list2tensor(inputs, shape):
        inputs = np.reshape(np.array(inputs).astype(np.int64), shape)
        return paddle.to_tensor(inputs)

    # 加载模型参数到模型中，设置为验证模式eval（）
    model_state_dict = paddle.load(params_file_path)
    model.load_dict(model_state_dict)
    model.eval()
    # 获得整个数据集的数据
    dataset = model.Dataset.dataset
    loader = model.Dataset.load_data(dataset=dataset, mode='', BATCHSIZE=1)
    for i, data in enumerate(loader()):
        # 获得用户数据，电影数据，评分数据  
        # 本案例只转换所有在样本中出现过的user和movie，实际中可以使用业务系统中的全量数据
        # usr_info, job_info, score = dataset[i]['usr_info'], dataset[i]['job_info'],dataset[i]['scores']
        usrid = str(dataset[i]['usr_info']['usr_id'])
        jobid = str(dataset[i]['job_info']['job_id'])
        usr_info, job_info, _ = data
        # 获得用户数据，计算得到用户特征，保存在usr_pkl字典中
        if usrid not in usr_pkl.keys():
            # usr_id_v = list2tensor(usr_info['usr_id'], [1])
            # usr_title_v = list2tensor(usr_info['title'], [39])
            # usr_state_v = list2tensor(usr_info['state'], [1])

            # usr_in = [usr_id_v, usr_title_v, usr_state_v, bert]
            # for var in usr_info:
            #     print(var)
            usr_in = [paddle.to_tensor(var) for var in usr_info]
            usr_feat = model.get_usr_feat(usr_in)

            usr_pkl[usrid] = usr_feat.numpy()
        
        # 获得电影数据，计算得到电影特征，保存在mov_pkl字典中
        if jobid not in job_pkl.keys():

            # job_id_v = list2tensor(job_info['job_id'], [1])
            # job_city_v = list2tensor(job_info['job_city'], [1, 1, 15])
            # job_exp_year_v = list2tensor(job_info['job_exp_year'], [1, 6])

            # job_in = [job_id_v, job_city_v, job_exp_year_v]
            job_in = [paddle.to_tensor(var) for var in job_info]

            job_feat = model.get_job_feat(job_in)

            job_pkl[jobid] = job_feat.numpy()
    


    print(len(job_pkl.keys()))
    print(len(usr_pkl.keys()))

    print(list(job_pkl.keys())[0], job_pkl[list(job_pkl.keys())[0]])
    print(list(usr_pkl.keys())[0], usr_pkl[list(usr_pkl.keys())[0]])

    # 保存特征到本地
    pickle.dump(usr_pkl, open('./usr_feat.pkl', 'wb'))
    pickle.dump(job_pkl, open('./job_feat.pkl', 'wb'))
    print("usr / job features saved!!!")
fc_sizes=[128, 64, 32]
model = Model(use_poster=False,use_usr_title=True, use_usr_state=True, use_job_city=True, use_job_exp_year=True,fc_sizes=fc_sizes)

param_path = "./checkpoint/epoch9.pdparams"
get_usr_job_features(model, param_path) 