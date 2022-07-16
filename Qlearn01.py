import numpy as np
import random
from tqdm import tqdm

import matplotlib.pyplot as plt
# %matplotlib inline

# is env game ให้เดินไปข้างหน้า
class MyGame:
    def __init__(self,point = 0,state = 0,max_state = 10):
        self.point = point
        self.state = state
        self.max_state = max_state
        # self.startGame()
    def gameOnProgress(self,value):
        # print("value::"+ str(value))
        # print("self.state::"+str(self.state))
        if(self.point == 10) or (self.state >= self.max_state - 1):
            return self.state, 0, True;
        else:
            self.state += 1;
            if(value == 0):
                # print("back")
                return self.state, 0, False;
            elif(value == 1):
                # print("forward")
                self.point += 1
                # print("point : "+ str(self.point))
                return self.state, 1, False;
        # self.startGame(value);
    def reset(self):
        self.point = 0;
        self.state = 0;
        return 0
        # self.startGame(value);

game = MyGame()

rewards = []
avg_rewards = []
total_step = []

# FUCK
# Q(s,a) = Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
lr           = 0.8 # learning rate อัตราการเรียนรู้
gamma        = 0.95 # discount rate 
num_episodes = 10000 # จำนวนรอบของเกมส์ที่เราจะเล่น
max_steps    = 99 # จำนวนครั้ง action ที่เลือกได้มากสุด ต่อ 1 เกมส์

epsilon      = 1.0 # ค่าการเรียนรู้ epsilon greedy
max_epsilon  = 1.0 # ใช้สำหรับอัพเดท epsilon 
min_epsilon  = 0.001 # ใช้สำหรับอัพเดท epsilon 
decay_rate   = 0.005  # ใช้สำหรับอัพเดท epsilon 

action_size  = 2 # จำนวน action ที่สามารถเลือกได้ 0 1
state_size   = 10 # จำนวน state ทั้งหมดในเกมส์
qtable       = np.zeros((state_size, action_size)) # ตาราง q-table

# print(random.uniform(0, 1))

for episode in tqdm(range(num_episodes)):
    state = game.reset() # state เริ่มต้น
    step = 0 # จำนวน action
    done = False # เอาไว้เช็คว่า เกมส์จบหรือยัง
    total_rewards = 0
    for step in range(max_steps):
        # print(np.argmax(qtable[state,:]))
        # ถ้าเรา random ช่วง 0-1 แล้วมีค่ามากกว่า epsilon เราจะ exploit (greedy)
        if random.uniform(0, 1) > epsilon: 
            action = np.argmax(qtable[state,:])
        else: # ถ้าได้น้อยกว่า เราจะ random เลือก action
            action = random.randint(0, 1)
            # print("action::"+str(action))
        # print(game.gameOnProgress(action))

        state_,reward, done = game.gameOnProgress(action)
        # print("state___",state_)
        # Q(s,a) = Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        qtable[state, action] = qtable[state, action] + lr * (reward + gamma * np.max(qtable[state_, :]) - qtable[state, action])
        
        total_rewards += reward
        # print("reward__",total_rewards)

        
        
        # เปลี่ยน state เป็น state หลังจาก action ไปแล้ว
        state = state_
        if (done):
            break
        # print("reward :"+str(reward))
     # ปรับ epsilon
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
    
    total_step.append(step)
    rewards.append(total_rewards)
    avg_rewards.append(total_rewards/(episode+1))

print ("Score over time: " +  str(sum(rewards)/num_episodes))
print(qtable)
# _ = plt.scatter(range(num_episodes), total_step)
        
