import numpy as np
from sklearn.preprocessing import StandardScaler
from dataset import Dataset

class Layer():
    def __init__(self, input_dim, output_dim, act, bias=True):
        self.bias = bias
        self.input = input_dim + 1 if bias else input_dim
        self.output = output_dim
        self.W = np.zeros([self.input, self.output])
        self.grad_W = None
        if act not in ['sigmoid', 'tanh', 'none']:
            assert(False)
        self.act = act

    def __call__(self, x, train=False):
        return self.forward(x, train)
    
    def forward(self, x, train):
        # x.shape: [bs, c]
        
        # TODO: 实现 y = Wx + b (bias通过增广W矩阵实现)
        pass

        if self.act=='sigmoid':
            # TODO: 实现 o = sigmoid(y)
            pass
        elif self.act == 'tanh':
            # TODO: 实现 o = tanh(y)
            pass
        elif self.act == 'none':
            # 不使用激活函数, 仅供参考
            o = y
        
        if train:
            # 模型训练时需要记录中间数据，用于反向梯度传播
            self.o = o
            self.x = x
        return o
    
    def backward(self, grad_o):
        if self.act=='sigmoid':
            # TODO: 计算o关于y的梯度, grad_y = o'(y) = sigmoid'(y)
            pass
        elif self.act == 'tanh':
            # TODO: 计算o关于y的梯度, grad_y = o'(y) = tanh'(y)
            pass
        elif self.act == 'none':
            # 不使用激活函数, 仅供参考
            grad_y = grad_o

        # TODO: 计算o关于W的梯度, grad_W = o'(W) = o'(y) * y'(W) （需要考虑矩阵相乘的形式）
        pass

        # TODO: 计算o关于x的梯度, grad_x = o'(x) = o'(y) * y'(x) （需要考虑矩阵相乘的形式）
        pass

        # 记录当前layer参数W的梯度，用于更新当前layer
        self.grad_W = grad_W

        # 返回关于输入x的梯度，用于更新前一层layer
        if self.bias:
            return grad_x[:,:-1]
        else:
            return grad_x
        
    def update(self, lr):
        self.W = self.W - self.grad_W * lr

class Model():
    def __init__(self, input_dim, hidden_dim, out_dim, hidden_layers=1):
        self.input_layer = Layer(input_dim, hidden_dim, 'sigmoid', bias=True)
        self.hidden_layers = [Layer(hidden_dim, hidden_dim, 'sigmoid', bias=True) 
                              for i in range(hidden_layers)]
        self.output_layer = Layer(hidden_dim, out_dim, 'none', bias=True)
        
    def __call__(self, x, train=False):
        return self.forward(x, train)
    
    def cross_entropy_loss(self, p, Y):
        loss = -Y * np.log(p)
        return np.sum(loss) / p.shape[0]

    def softmax(self, logit):
        exponent = np.exp(logit)
        exponent_sum = np.sum(exponent, axis=-1, keepdims=True)
        return exponent / exponent_sum
    
    def softmax_backward(self, p, Y):
        return p - Y
    
    def forward(self, x, train):
    	# 前向传播
        h = self.input_layer(x, train)
        for hidden_layer in self.hidden_layers:
            h = hidden_layer(h, train)
        o = self.output_layer(h, train)
        p = self.softmax(o)
        return p
    
    def backward(self, p, Y):
    	# 反向传播
        grad = self.softmax_backward(p, Y)
        grad = self.output_layer.backward(grad)
        for hidden_layer in self.hidden_layers[::-1]:
            grad = hidden_layer.backward(grad)
        grad = self.input_layer.backward(grad)
    
    def update(self, lr):
        self.input_layer.update(lr)
        for hidden_layer in self.hidden_layers:
            hidden_layer.update(lr)
        self.output_layer.update(lr)

if __name__ == "__main__":

    # build dataset
    data = Dataset("./celeba")
    X_train, Y_train = data.get_train_data()
    X_test, Y_test = data.get_test_data()

    # 标准化训练集和测试集
    sc = StandardScaler()   # 定义一个标准缩放器
    sc.fit(X_train)         # 计算均值、标准差
    X_train = sc.transform(X_train) # 使用计算出的均值和标准差进行标准化
    X_test  = sc.transform(X_test)  # 使用计算出的均值和标准差进行标准化
    
    # build model
    N, C = X_train.shape
    input_dim = 40
    hidden_dim = 256
    output_dim = 10
    hidden_layers = 1
    model = Model(input_dim, hidden_dim, output_dim, hidden_layers)

    # train
    epochs = 4000
    batch_size = 50
    lr = 1e-2
    X = X_train.reshape([N//batch_size, batch_size, -1])
    Y = np.zeros((N, 10))
    for i in range(N):
        Y[i, Y_train[i]] = 1
    Y = Y.reshape([N//batch_size, batch_size, -1])
    loss_list = []
    print("Training Started")
    for i in range(epochs):
        for j in range(X.shape[0]):
            x, y = X[j], Y[j]
            p = model.forward(x, train=True)
            loss = model.cross_entropy_loss(p, y)
            model.backward(p, y)
            model.update(lr)
            loss_list.append(loss)
            # print(f"epoch: {i}  iter: {j}   loss: {loss}")
        if i % 50 == 0:
            N, C = X_train.shape
            X_train = X_train.reshape(N, -1)
            p = model.forward(X_train, train=False)
            pred = np.argmax(p, axis=1)
            acc = np.sum((pred==Y_train)) / N
            print(f"epoch: {i}  loss: {np.mean(loss_list)}  Train Acc: {acc}")
        if i == int(epochs*0.25):
            lr = lr / 2
        if i == int(epochs*0.5):
            lr = lr / 2
        if i == int(epochs*0.8):
            lr = lr / 2
        if i == int(epochs*0.9):
            lr = lr / 2
        loss_list = []
    print("Training Finished")

    # test
    N, C = X_test.shape
    X_test = X_test.reshape(N, -1)
    p = model.forward(X_test, train=False)
    pred = np.argmax(p, axis=1)
    acc = np.sum((pred==Y_test)) / N
    print(pred, Y_test)
    print(f"Test Accuracy: {acc}")