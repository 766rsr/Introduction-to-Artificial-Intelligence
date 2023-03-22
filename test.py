import numpy as np
import matplotlib.pyplot as plt

#----------------------------------------------
#  定義一個副函式進行線性回歸模型運算
#----------------------------------------------
def linear_regression(X,y,lr=0.1,n_iterations=10):

    #宣告並定義初始值
    b0=0
    w1=0    
    
    loss_history = []
    b0_history = []
    w1_hostory = []

    for i in range(n_iterations):

        #定義回歸模型
        y_pred = b0+w1*X
        loss = np.mean((y_pred-y)**2)

        #梯度下降之運算
        grad_b0 = np.mean(2*(y_pred-y))
        grad_w1 = np.mean(2*X*(y_pred-y))
        b0=b0-lr*grad_b0
        w1=w1-lr*grad_w1

        #儲存計算結果
        loss_history.append(loss)
        b0_history.append(b0)
        w1_hostory.append(w1)

    return b0,w1,loss_history,b0_history,w1_hostory #回傳結果


#----------------------------------------------
#  隨機生成一測試資料並調用上方定義之副函式
#----------------------------------------------
np.random.seed(0)
n_samples = 100
X = np.random.randn(n_samples)
y = 2*X+1+np.random.randn(n_samples)

w0,w1,loss_history,w0_history,w1_history= linear_regression(X,y) #呼叫模型


#----------------------------------------------
#  繪製測試結果之圖表
#----------------------------------------------
plt.figure(figsize=(10,5))
plt.plot(loss_history)
plt.xlabel('Iteration')
plt.ylabel('MES Loss')
plt.title('Loss Thrend')

plt.figure(figsize=(10,5))
plt.plot(w0_history)
plt.xlabel('Iteration')
plt.ylabel('w0')
plt.title('w0 Thrend')

plt.plot(w1_history)
plt.xlabel('Iteration')
plt.ylabel('W1')
plt.title('w1 Tread')


plt.figure(figsize=(10,5))
plt.scatter(X,y)
plt.plot(X,w0 + w1*X, color='r')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression')

plt.show()