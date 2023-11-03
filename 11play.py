import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import cma
import numpy as np
import os
import cv2
from torchvision import transforms
from PIL import Image
import cv2

# 创建 VideoWriter 对象
output_filename = '11output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码器
fps = 25  # 帧率
video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (960, 960))

def transform_image(input_image, image_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    # 将输入图像转换为PIL图像对象
    pil_image = Image.fromarray(input_image)
    # 应用转换操作
    transformed_image = transform(pil_image)
    return transformed_image

# 定义神经网络模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self._hidden_size=hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self._hidden=torch.zeros((1, 1,hidden_size)).cuda()
        self._cell=torch.zeros((1, 1,hidden_size)).cuda()
    def forward(self, x):
        out, (self._hidden, self._cell)= self.lstm(x.view(1,1,-1),(self._hidden, self._cell))
        #print("out.shape:",out.shape)
        #print("hidden.shape:",hidden.shape)
        #print("cell.shape:",cell.shape)
        out=torch.flatten(out,start_dim=1,end_dim=-1)
        out = self.fc(out)
        out = torch.sigmoid(out)    #使用torch.sigmoid将输出缩放到0到output_size之间
        return out
    def reset(self):
        self._hidden=torch.zeros((1, 1,self._hidden_size)).cuda()
        self._cell=torch.zeros((1, 1,self._hidden_size)).cuda()

class SelfAttention(nn.Module):
    """A simple self-attention solution."""

    def __init__(self, data_dim, dim_q):
        super(SelfAttention, self).__init__()
        self._layers = []

        self._fc_q = nn.Linear(data_dim, dim_q)
        self._layers.append(self._fc_q)
        self._fc_k = nn.Linear(data_dim, dim_q)
        self._layers.append(self._fc_k)

    def forward(self, input_data):
        # Expect input_data to be of shape (b, t, k).
        b, t, k = input_data.size()

        # Linear transforms.
        queries = self._fc_q(input=input_data)  # (b, t, q)
        keys = self._fc_k(input=input_data)  # (b, t, q)

        # Attention matrix.
        dot = torch.bmm(queries, keys.transpose(1, 2))  # (b, t, t)
        scaled_dot = torch.div(dot, torch.sqrt(torch.tensor(k).float()))
        return scaled_dot

# Merging the networks together
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,data_dim, dim_q):
        super(NeuralNetwork, self).__init__()
        self.SelfAttention = SelfAttention(data_dim, dim_q)
        self.LSTMModel = LSTMModel(input_size, hidden_size, output_size)

    def forward(self, x):
        x = self.SelfAttention(x)
        x = self.LSTMModel(x)
        return x

    def reset(self):
        self.LSTMModel.reset()


# 定义目标函数
def evaluate(env):
    state_dict = {}
    start = 0
    model.reset()
    total_reward = 0
    count=0
    with torch.no_grad():
        for iseed in range(repeat_num):  # 重复执行repeat_num次
            observation,info= env.reset(seed=iseed+SEED)
            episode_reward = 0
            while True:
                task_image = observation
                #observation=observation[:image_size,10:image_size-10,:]
                #print("ob shape1:",observation.shape)                                        #96 96 3
                #observation = torch.from_numpy(observation).float().cuda()       #gpu
                observation = transform_image(observation,image_size).permute(1, 2, 0).cuda()
                h, w, c = observation.size()
                #print(f'h:{h},w:{w},c:{c}')                                                                #96 96 1
                patches = observation.unfold(0, patch_size, patch_stride).permute(0, 3, 1, 2)
                patches = patches.unfold(2, patch_size, patch_stride).permute(0, 2, 1, 4, 3)
                patches = patches.reshape((-1, patch_size, patch_size, c))
                #print("patches.shape:",patches.shape)                                          #225 7 7 1
                # flattened_patches.shape = (1, n, p * p * c)
                flattened_patches = patches.reshape((1, -1, c * patch_size ** 2))
                #print("flattened_patches.shape:",flattened_patches.shape)           #1 225 49
                # attention_matrix.shape = (1, n, n)
                attention_matrix = model.SelfAttention(flattened_patches)
                #print("attention_matrix.shape:",attention_matrix.shape)                #1 225 225
                # patch_importance_matrix.shape = (n, n)
                patch_importance_matrix = torch.softmax(attention_matrix.squeeze(), dim=-1)
                # patch_importance.shape = (n,)
                patch_importance = patch_importance_matrix.sum(dim=0)
                #print("patch_importance.shape:",patch_importance.shape)           #255
                # extract top k important patches
                ix = torch.argsort(patch_importance, descending=True)
                #print("ix:",ix)
                top_k_ix = ix[:top_k].cpu()
                #print("top_k_ix:",top_k_ix)
                centers = patch_centers[top_k_ix]
                #print("centers.shape1:",centers.shape)                                                  #10 2
                centers1=centers
                #print("patch_centers.shape:",patch_centers)    
                centers = centers.flatten(0, -1)                                                              #展平[20]
                centers = centers / image_size
                centers = centers.unsqueeze(0).cuda()
                #print("centers.shape2:",centers.shape) 
                #"""
                # Overplot.
                if show_overplot:
                    #task_image = observation.cpu().numpy().copy()
                    #cv2.imshow('Overplotting1', task_image[:, :,[2,1,0]])
                    #cv2.waitKey(1)
                    #print("task_image:",task_image.shape)

                    patch_importance_copy = patch_importance.cpu().numpy().copy()
                    #print("patch_importance_copy:",patch_importance_copy.shape)

                    white_patch = np.ones((patch_size, patch_size, 3))
                    #white_patch = np.zeros((patch_size, patch_size, 3))
                    #print("white_patch:",white_patch.shape)

                    half_patch_size = patch_size // 2
                    #print("half_patch_size:",half_patch_size)
                    #print("centers1:",centers1.shape)
                    for i, center in enumerate(centers1):
                        row_ss = int(center[0]) - half_patch_size
                        row_ee = int(center[0]) + half_patch_size + 1
                        col_ss = int(center[1]) - half_patch_size
                        col_ee = int(center[1]) + half_patch_size + 1
                        ratio = 1.0 * i / top_k
                        #print(f'center[0]:{center[0]},center[1]:{center[1]},center:{center}')
                        #print(f'row_ss:{row_ss},row_ee:{row_ee},col_ss:{col_ss},col_ee:{col_ee},ratio:{ratio}')
                        #task_image[row_ss:row_ee, col_ss:col_ee] = (1-ratio) * white_patch
                        task_image[row_ss:row_ee, col_ss:col_ee] =(ratio * task_image[row_ss:row_ee, col_ss:col_ee] + (1-ratio) * white_patch*255)
                        """
                        if i%2==1:
                            task_image[row_ss:row_ee, col_ss:col_ee, 1] = 255  # 将绿色通道设为255
                            task_image[row_ss:row_ee, col_ss:col_ee, 2] = 255  # 将红色通道设为255
                            task_image[row_ss:row_ee, col_ss:col_ee, 3] = 128  # 将红色通道设为255
                        else:
                            task_image[row_ss:row_ee, col_ss:col_ee, 1] = 255  # 将绿色通道设为255
                            task_image[row_ss:row_ee, col_ss:col_ee, 0] = 255  # 将红色通道设为255
                            task_image[row_ss:row_ee, col_ss:col_ee, 3] = 128  # 将红色通道设为255
                        """
                    task_image = cv2.resize(task_image, (task_image.shape[0] * 10, task_image.shape[1] * 10))
                    #print("task_image:",task_image.shape[0])
                    cv2.imshow('Overplotting', task_image[:, :,[2,1,0]])
                    video_writer.write(task_image[:, :,[2,1,0]])
                    cv2.waitKey(1)

                #"""
                #print("centers.shape2:",centers.shape)                                                  #20
                action = model.LSTMModel(centers)
                #print("action1:",action)
                #action = int(action)
                #action = torch.argmin(action).item()
                #print("action2:",action)
                action = action.cpu().detach().numpy().squeeze() 
                action[0]=action[0]*2-1
                #action = action.cpu().detach().numpy().squeeze() 
                #print("action:",action)
                observation, reward, t1, t2 ,_= env.step(action)
                #observation=np.array(observation)
                episode_reward += reward
                done = (t1 or t2)
                if reward<0:
                   count=count+1
                else:
                   count=0
                if count>n_count_num:
                   done=True
                if done:
                    break

            total_reward += episode_reward
            print(f'seed:,{iseed+SEED},reward:{episode_reward},i:{iseed},SEED:{SEED}')
    average_reward = total_reward / repeat_num  # 计算平均奖励
    print("average_reward :",average_reward )
    # 释放 VideoWriter 对象
    video_writer.release()
    return -average_reward  # 返回负的平均奖励


# 设置超参数
top_k=6
input_size = top_k*2
hidden_size = 8
output_size = 3
population_size = 1
sigma = 0.41
num_iterations = 1
image_size=76
patch_size=7
patch_stride=6
data_dim=patch_size*patch_size*3
dim_q=4
#show_overplot=True
show_overplot=False
screen_dir=""
img_ix = 1
repeat_num=1  #重复次数
SEED=3117     #赛道种子
n_count_num=20  #负分次数
#图像初始参数
n = int((image_size - patch_size) / patch_stride + 1)
offset = patch_size // 2
patch_centers = []
for i in range(n):
     patch_center_row = offset + i * patch_stride
     for j in range(n):
          patch_center_col = offset + j * patch_stride
          patch_centers.append([patch_center_row, patch_center_col])
patch_centers = torch.tensor(patch_centers).float()
num_patches = n ** 2
print('num_patches = {}'.format(num_patches))
print("patch_centers.shape:",patch_centers.shape)
def print_population_size(cmaes):
    population_size = len(cmaes.ask())
    print(f"Generation {cmaes.countiter + 1}: Population Size = {population_size}")


# 创建CartPole-v1环境和神经网络模型
#env = gym.make("CarRacing-v2",render_mode="rgb_array",continuous=False)
#env = gym.make("CarRacing-v2",render_mode="human",continuous=False)
eval_env = gym.make("CarRacing-v2",render_mode="human", continuous=True)

model = NeuralNetwork(input_size, hidden_size, output_size,data_dim, dim_q)
model.load_state_dict(torch.load('11941.pth'))
print("load 11.pth")
model.cuda()

# 在环境中评估最优解
show_overplot=True
n_count_num=20  #负分次数
total_rewards = []
for _ in range(repeat_num):
    reward = evaluate(eval_env)
    total_rewards.append(reward)
    print('reward:', reward)
mean_reward = np.mean(total_rewards)
print('Mean reward:', mean_reward)

# 关闭环境
eval_env.close()