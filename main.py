import numpy as np
import paddle
import paddle.nn.functional as F
from model import Model
from math import sqrt


def evaluation(model):
    # model_state_dict = paddle.load(params_file_path)
    # model.load_dict(model_state_dict)
    model.eval()

    acc_set = []
    avg_loss_set = []
    squaredError=[]
    for idx, data in enumerate(model.valid_loader()):
        usr, mov, score_label = data
        usr_v = [paddle.to_tensor(var) for var in usr]
        mov_v = [paddle.to_tensor(var) for var in mov]

        _, _, scores_predict = model(usr_v, mov_v)
        pred_scores = scores_predict.numpy()
        
        avg_loss_set.append(np.mean(np.abs(pred_scores - score_label)))
        squaredError.extend(np.abs(pred_scores - score_label)**2)

        diff = np.abs(pred_scores - score_label)
        diff[diff>0.5] = 1
        acc = 1 - np.mean(diff)
        acc_set.append(acc)
    RMSE=sqrt(np.sum(squaredError) / len(squaredError))
    
    # print("RMSE = ", sqrt(np.sum(squaredError) / len(squaredError)))#均方根误差RMSE
        
    return np.mean(acc_set), np.mean(avg_loss_set),RMSE

def train(model):
    # 配置训练参数
    lr = 5e-4
    Epoches = 20
    paddle.set_device('cpu') 

    # 启动训练
    model.train()
    # 获得数据读取器
    data_loader = model.train_loader
    # 使用adam优化器，学习率使用0.01
    opt = paddle.optimizer.Adam(learning_rate=lr, parameters=model.parameters())
    
    for epoch in range(0, Epoches):
        for idx, data in enumerate(data_loader()):
            # 获得数据，并转为tensor格式
            usr, mov, score = data
            usr_v = [paddle.to_tensor(var) for var in usr]
            mov_v = [paddle.to_tensor(var) for var in mov]
            scores_label = paddle.to_tensor(score)
            # 计算出算法的前向计算结果
       
            _, _, scores_predict = model(usr_v, mov_v)
            # 计算loss
            loss = F.square_error_cost(scores_predict, scores_label)
            avg_loss = paddle.mean(loss)

            if idx % 10 == 0:
                print("epoch: {}, batch_id: {}, loss is: {}".format(epoch, idx, avg_loss.numpy()))
                
            # 损失函数下降，并清除梯度
            avg_loss.backward()
            opt.step()
            opt.clear_grad()

        # 每个epoch 保存一次模型
        acc, mae,RMSE = evaluation(model)
        print("no:",epoch,"ACC:", acc, "MAE:", mae,'RMSE:',RMSE)
        model.train()
        paddle.save(model.state_dict(), './checkpoint/epoch'+str(epoch)+'.pdparams')

fc_sizes=[128, 64, 32]
model = Model(use_poster=False,use_usr_title=True, use_usr_state=True, use_job_city=True, use_job_exp_year=True,fc_sizes=fc_sizes)
train(model)

# param_path = "./checkpoint/epoch"
# for i in range(20):
#     acc, mae,RMSE = evaluation(model, param_path+str(i)+'.pdparams')
#     print("no:",i,"ACC:", acc, "MAE:", mae,'RMSE:',RMSE)