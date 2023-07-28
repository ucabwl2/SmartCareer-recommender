import streamlit as st
import pandas as pd
import pickle
import paddle
from Constants import Data_path
import numpy as np
from SimCSE import SimcseModel
from sklearn.metrics.pairwise import cosine_similarity
import streamlit.components.v1 as components
from skill_graph_html import desc_to_html, gen_query, draw

st.title('SmartCareer: Job and Candidate Matcher')
st.header('Enter the parameters here:')

job_df = pd.read_excel('./data/jobs.xlsx', sheet_name='summary', index_col='jobid')
job_df['res_col'] = job_df['jobtitle']+ '--' + job_df['Teaser']

usr_df = pd.read_excel("./data/freelancers.xlsx", sheet_name='freelancer summary', index_col='FreelancerID')
usr_df.to_excel("freelancer_sum_only.xlsx")
usr_df = pd.read_excel("freelancer_sum_only.xlsx", index_col='FreelancerID')
usr_df['res_col'] = usr_df['Title'].fillna('EMPTY TITLE').astype(str)+ '--' + usr_df['Summary'].fillna('EMPTY SUMMARY').astype(str)
fl_feats = pickle.load(open('usr_feat.pkl', 'rb'))
job_feats = pickle.load(open('job_feat.pkl', 'rb'))
usr_id = st.number_input('Job/User ID:', min_value=0)
description = st.text_area("Don't have an ID? Describe it (Optional)", "")
top_k = st.number_input('Top k items:', min_value=0)
pick_num = st.number_input('Randomly sample m items (m <=k ):', min_value=0)
item = st.radio(
    "What are you looking for",
    ('Jobs', 'Candidates'))



def recommend_job_for_usr(item, description, usr_id, top_k, pick_num):
    if item == 'Jobs': 
        usr_feats, mov_feats, mov_info_path, desc_emb_path, key_col, summary_col = [fl_feats, job_feats,'./data/jobs.xlsx', "./data/usr_sum_emb.json", 'FreelancerID', 'Summary_embedding']
    else:
        usr_feats, mov_feats, mov_info_path, desc_emb_path, key_col, summary_col = [job_feats, fl_feats,"./data/freelancers.xlsx", 
                                                                                            "./data/all_job_ProjectDescription_embedding.json",'jobid', 'ProjectDescription_embedding']
    assert pick_num <= top_k
    # cold start
    if description != "":
        if item == 'Jobs':
            desc_emb = pd.read_json(desc_emb_path)
            feat_ids = [int(i) for i in list(usr_feats.keys())]
            desc_emb=desc_emb.set_index(key_col).loc[feat_ids]
            model = SimcseModel()
            cur_emb = model(description)
            all_emb = np.array(desc_emb[summary_col].to_list())
            desc_emb['similarity'] = cosine_similarity(cur_emb, np.array(all_emb))[0]
            usr_id = desc_emb.sort_values("similarity").index.to_list()[-1]
        else:
            desc_emb = pd.read_json(desc_emb_path)
            avg_emb = [np.array([i[0] for i in desc_emb[summary_col].dropna().values]).mean(axis=0).tolist()]
            desc_emb[summary_col] = desc_emb[summary_col].apply(lambda d: d if isinstance(d, list) else avg_emb)
            feat_ids = [int(i) for i in list(usr_feats.keys())]
            desc_emb=desc_emb.set_index(key_col).loc[feat_ids]
            model = SimcseModel()
            cur_emb = model(description)
            all_emb = np.array(desc_emb[summary_col].apply(lambda x:x[0]).to_list())
            desc_emb['similarity'] = cosine_similarity(cur_emb, np.array(all_emb))[0]
            usr_id = desc_emb.sort_values("similarity").index.to_list()[-1]
        desc_to_html(description, "demo_net.html")

    else:
        if item == 'Jobs':
            cypher = gen_query(item, usr_id, pick_num)
        else:
            title = job_df.loc[usr_id]['jobtitle']
            cypher = gen_query(item, title, pick_num)
        draw(cypher, "demo_net.html")
    HtmlFile = open("demo_net.html", 'r', encoding='utf-8')
    components.html(HtmlFile.read(), height=435)
    
    # 读取电影和用户的特征

    usr_feat = usr_feats[str(usr_id)]

    cos_sims = []

    # with dygraph.guard():
    paddle.disable_static()
    # 索引电影特征，计算和输入用户ID的特征的相似度
    for idx, key in enumerate(mov_feats.keys()):
        mov_feat = mov_feats[key]
        usr_feat = paddle.to_tensor(usr_feat)
        mov_feat = paddle.to_tensor(mov_feat)
        # 计算余弦相似度
        sim = paddle.nn.functional.common.cosine_similarity(usr_feat, mov_feat)
        
        cos_sims.append(sim.numpy()[0])
    # 对相似度排序
    index = np.argsort(cos_sims)[-top_k:]

    # 读取电影文件里的数据，根据电影ID索引到电影信息
    if item=='Jobs':
        df = job_df
    else:
        df = usr_df
    mov_info = df['res_col'].to_dict()
    print("For this user:")
    print("usr_id:", usr_id)
    print("Possible Matching Jobs Are")
    res = []
    str_res = []
    # 加入随机选择因素，确保每次推荐的都不一样
    while len(res) < pick_num:
        val = np.random.choice(len(index), 1)[0]
        idx = index[val]
        mov_id = list(mov_feats.keys())[idx]
        if mov_id not in res:
            res.append(mov_id)
    print('========',res)
    for id in res:
        print("job_id:", id, mov_info[int(id)])
        str_res.append(f"""id: {id}  {mov_info[int(id)]} \n """) 
    return str_res

if st.button('Recommend For Me'):
    str_res = recommend_job_for_usr(item, description, usr_id, top_k ,pick_num)
    if item=='Jobs':
        st.success(f"""
        For this user \n
        usr_id: {usr_id} \n
        possible matching jobs are \n
        """)
    else:
        st.success(f"""
        For this job \n
        job_id: {usr_id} \n
        possible matching candidates are \n
        """)
    for res in str_res:
        st.write(res)
        # 1 <-> 7530
        # "Guidewire QA/Tester Lead (TE-7365)"