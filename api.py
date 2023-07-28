from flask import Flask, request, render_template,jsonify
import pickle 
import numpy as np
import paddle
import pandas as pd
from Constants import Data_path
app = Flask(__name__)

# def do_something(text1,text2):
#    text1 = text1.upper()
#    text2 = text2.upper()
#    combine = text1 + text2
#    return combine

# 定义根据用户兴趣推荐电影
def recommend_mov_for_usr(usr_id, top_k ,pick_num):
    usr_feat_dir, mov_feat_dir, mov_info_path = 'usr_feat.pkl', 'job_feat.pkl',"./data/jobs.xlsx"
    assert pick_num <= top_k
    # 读取电影和用户的特征
    usr_feats = pickle.load(open(usr_feat_dir, 'rb'))
    mov_feats = pickle.load(open(mov_feat_dir, 'rb'))
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
    df = pd.read_excel(Data_path+"jobs.xlsx",sheet_name='summary', index_col='jobid')
    df['res_col'] = df['jobtitle']+ '--' + df['Teaser']
    mov_info = df['res_col'].to_dict()
    print("For this user:")
    print("usr_id:", usr_id)
    print("Possible Matching Jobs Are")
    res = []
    str_res = f"""
    "For this user:" \n
    "usr_id:", {usr_id} \n
    "Possible Matching Jobs Are" \n
    """
    # 加入随机选择因素，确保每次推荐的都不一样
    while len(res) < pick_num:
        val = np.random.choice(len(index), 1)[0]
        idx = index[val]
        mov_id = list(mov_feats.keys())[idx]
        if mov_id not in res:
            res.append(mov_id)
    print('========',res)
    for id in res:
        print("mov_id:", id, mov_info[int(id)])
        str_res += f"mov_id: {id}  {mov_info[int(id)]} \n "
    return str_res

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/join', methods=['GET','POST'])
def my_form_post():
    usr_id = int(request.form['text1'])
    top_k = int(request.form['text2'])
    pick_num = int(request.form['text3'])
    combine = recommend_mov_for_usr(usr_id, top_k,pick_num)
    result = {
        "output": combine
    }
    print(result)
    result = {str(key): value for key, value in result.items()}
    return jsonify(result=result)

if __name__ == '__main__':
    app.run(host='0.0.0.0')

