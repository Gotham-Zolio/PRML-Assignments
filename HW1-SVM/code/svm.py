import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from dataset import Dataset

# 构建数据集
data = Dataset("../celeba")
X_imgs_train, X_attrs_train, Y_train = data.get_train_data()
X_imgs_test, X_attrs_test, Y_test = data.get_test_data()

# 选择属性特征作为分类依据
X_train = X_attrs_train
X_test = X_attrs_test

# TODO: 选择原始图片作为分类依据
# 提示: 单张图片的原始形状是[H, W, 3]，需将其转换成一维特征再利用SVM分类
X_train = X_imgs_train.reshape(X_imgs_train.shape[0], -1)
X_test = X_imgs_test.reshape(X_imgs_test.shape[0], -1)

# 标准化训练集和测试集
sc = StandardScaler()               # 定义一个标准缩放器
sc.fit(X_train)                     # 计算均值、标准差
X_train_std = sc.transform(X_train) # 使用计算出的均值和标准差进行标准化
X_test_std  = sc.transform(X_test)  # 使用计算出的均值和标准差进行标准化

# TODO: 训练支持向量机
# 提示: 
svm = SVC(kernel = "linear", C = 2.4771e-05)        # 定义线性支持向量分类器 (linear为线性核函数)
svm.fit(X_train_std, Y_train)       # 根据给定的训练数据拟合训练SVM模型

# 使用测试集进行数据预测
Y_pred = svm.predict(X_test_std)    # 用训练好的分类器svm预测数据X_test_std的标签
print('Misclassified samples: %d' % (Y_test != Y_pred).sum())   # 输出错误分类的样本数
print('Accuracy: %.2f' % svm.score(X_test_std, Y_test))         # 输出分类准确率