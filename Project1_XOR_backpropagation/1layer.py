# coding: utf-8

# 인공지능(딥러닝)개론 # Homework 1
# 간단한 XOR Table을 학습하는 NN을 구성하는 문제입니다.
# 
#  1-Layer, 2-Layer model을 각각 구성하여 XOR 결과를 비교
#  1-Layer, 2-Layer의 model을 feedforward network와 Backpropagation을 이용하여 학습


## 이 외에 추가 라이브러리 사용 금지
import numpy as np
import random
import matplotlib.pyplot as plt



# Hyper parameters
## 학습의 횟수와 Gradient update에 쓰이는 learning rate
## 다른 값을 사용하여도 무방합니다.
epochs = 10000
learning_rate = 0.05



# Input data setting
## XOR data 
## 입력 데이터들, XOR Table에 맞게 정의
train_inp = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
train_out = np.array([0, 1, 1, 0])



# Weight Setting
## 학습에 사용되는 weight들의 초기값을 선언
## 현재 weight변수는 2-layer 기준으로 설정
W1 = np.random.randn(2,1)
#W2 = np.random.randn(2,1)
b1 = np.random.randn(1,1)
#b2 = np.random.randn(1,1)

##-----------------------------------##
##------- Activation Function -------##
##-----------------------------------##
def tanh(x):
    numerator = np.exp(x) - np.exp(-x)
    denominator = np.exp(x) + np.exp(-x)
    return numerator/denominator
def dtanh(x):
    return (1-tanh(x))*(1+tanh(x))

# ----------------------------------- #
# --------- Training Step ----------- #
# ----------------------------------- #
# 학습이 시작
# epoch 사이즈만큼 for 문을 돌며 학습

errors = []
for epoch in range(epochs):
        
    # 데이터 4가지 중 랜덤으로 하나 선택
    for batch in range(4):
        idx = random.randint(0,3)

        # 입력 데이터 xin과 해당하는 정답 ans 불러오기
        xin = train_inp[idx].reshape(1,2)
        ans = train_out[idx]
        
        # Layer에 맞는 Forward Network 구성
        net1 = np.matmul(xin,W1)+b1
        
        ###activation function
        fnet1=tanh(net1)
        

        # Mean Squared Error (MSE)로 loss 계산
        loss = pow(ans-fnet1,2)        
        
        # delta matrix initialization(Zero 값이 아닌 다른 방법으로 이용하셔도 됩니다.)
        delta_W1 = np.zeros((2,1))
        #delta_W2 = np.zeros((2,1))
        delta_b1 = np.zeros((1,1))
        #delta_b2 = np.zeros((1,1))
        
        
        # Backpropagation을 통한 Weight의 Gradient calculation(update)
        delta_W1 = -(ans-fnet1)*dtanh(net1)*xin.T
        delta_b1 = -(ans-fnet1)*dtanh(net1)      
        

        # 각 weight의 update 반영
        W1 = W1 - learning_rate * delta_W1
        b1 = b1 - learning_rate * delta_b1
        
            
    ## 500번째 epoch마다 loss를 프린트
    if epoch%500 == 0:
        print("epoch[{}/{}] loss: {:.4f}".format(epoch,epochs,float(loss)))
        
    ## plot을 위해 값 저장
    errors.append(loss)



## 학습이 끝난 후, loss를 확인
loss =  np.array(errors)
plt.plot(loss.reshape(epochs))
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()



#-----------------------------------#
#--------- Testing Step ------------#
#-----------------------------------#

for idx in range(4):
    xin = train_inp[idx]
    ans = train_out[idx]
    
    net1 = tanh(np.matmul(xin,W1)+b1)
    pred = net1 # ans와 가까울 수록 잘 학습된 것을 의미
    
    print("input: ", xin, ", answer: ", ans, ", pred: {:.4f}".format(float(pred)))
    


#-----------------------------------#
#--------- Weight Saving -----------#
#-----------------------------------#

# weight, bias를 저장하는 부분

np.savetxt("20171875_layer1_weight.txt",(W1, b1),fmt="%s")
