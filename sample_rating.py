import random
import numpy as np
import pandas as pd
from PIL import Image

rating_path = "./data/rating.xlsx"
job_path = "./data/jobs.xlsx"
df_rating = pd.read_excel(open(rating_path, 'rb'),
                             sheet_name='shortlists')
df_job = pd.read_excel(open(job_path, 'rb'),
                             sheet_name='summary')['jobid']
jobid = df_job['jobid'].values

# df_rating[df_rating['FreelancerID']]
count=1
dic_1 = {}
dic_2 = {}
list = []

for i in range(len(df_rating)-1):
        if df_rating['FreelancerID'][i] == df_rating['FreelancerID'][i+1]:
            list.append(df_rating['jobid'][i])
            count+=1


        else:
            list.append(df_rating['jobid'][i])
            dic_1[df_rating['FreelancerID'][i]] = count
            dic_2[df_rating['FreelancerID'][i]] = list
            list = []
            count =1

rating = pd.read_excel(open(rating_path, 'rb'),sheet_name='shortlists') ## read in rating file
job_df = pd.read_excel(open(job_path, 'rb'),sheet_name='summary')

def get_rating(rating_path, job_path):
    rating = pd.read_excel(open(rating_path, 'rb'),sheet_name='shortlists') ## read in rating file
    job_df = pd.read_excel(open(job_path, 'rb'),sheet_name='summary')
    all_job_set = set(job_df['jobid'].values)
    rating_dict = rating.to_dict('records')   ## change to dictionary

    tmp_dic = {}
    for ele in rating_dict:
        if ele['FreelancerID'] not in tmp_dic:
            tmp_dic[ele['FreelancerID']] = [ele['jobid']]
        else:
            tmp_dic[ele['FreelancerID']].append(ele['jobid'])
    print(f'We have {len(tmp_dic)} unique freelancers')
    
    res = pd.DataFrame()
    active_job_cnt_all = 0
    for userid,job in tmp_dic.items():

        active_job_cnt = len(job)
        active_job_cnt_all += active_job_cnt
        
        
        inactive_jobs = all_job_set - set(job)
        job.extend(np.random.choice(list(inactive_jobs), size=len(job), replace=False).tolist())
        tmp_res = {}
        tmp_res['FreelancerID'] = userid
        tmp_res['jobid'] = job
        tmp_res['rating'] = [1] * active_job_cnt + [0] * active_job_cnt
        tmp_res_df = pd.DataFrame.from_dict(tmp_res)
        res = res.append(tmp_res_df,ignore_index=True)
    print(f'There are {active_job_cnt_all} active jobs, and output data shape is {res.shape}')
    return res

result = get_rating(rating_path,job_path)
result.to_excel('./data/post_rating.xlsx',index=False)