# coding:utf-8

from math import log
import operator
import treePlotter
from collections import Counter
import sys
import copy
import os
from treeFunc import *


def cal_entropy(dataset):
    """
    计算给定数据集的信息熵。

    Parameters:
    dataset (list): 数据集，每个元素是一个列表，表示一个样本的特征和标签（最后一个元素）。

    Returns:
    float: 计算得到的信息熵。
    """
    numEntries = len(dataset)  # 数据集中样本的总数
    labelCounts = {}  # 初始化标签计数字典

    # 统计每个类别的样本数量
    for featVec in dataset:
        currentLabel = featVec[-1]  # 获取样本的标签（最后一个元素）
        if currentLabel not in labelCounts:
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

    entropy = 0.0

    # 计算信息熵
    for count in labelCounts.values():
        # TODO
        prob = count / numEntries
        entropy += -prob * log(prob, 2)

    return entropy


def ID3_chooseBestFeatureToSplit(dataset):
    """
    使用ID3算法选择最佳划分特征。

    Parameters:
    dataset (list): 数据集，每个元素是一个列表，表示一个样本的特征和标签（最后一个元素）。

    Returns:
    int: 最佳划分特征的索引。
    """
    numFeatures = len(dataset[0]) - 1  # 特征的数量（不包括标签）
    baseEnt = cal_entropy(dataset)  # 计算数据集的基础信息熵
    bestInfoGain = 0.0  # 最佳信息增益初始化
    bestFeature = -1  # 最佳划分特征初始化

    # 遍历所有特征
    for i in range(numFeatures):
        featList = [example[i] for example in dataset]  # 获取第i个特征的所有取值
        uniqueVals = set(featList)  # 将特征值列表转换为集合，得到唯一的特征取值

        newEnt = 0.0  # 初始化新的信息熵
        # 计算每种划分方式的信息熵
        for value in uniqueVals:
            subdataset = splitdataset(dataset, i, value)  # 根据特征i和取值value划分子数据集
            # TODO
            prob = len(subdataset) / len(dataset)
            newEnt += prob * cal_entropy(subdataset)

        infoGain = baseEnt - newEnt  # 计算信息增益
        print("ID3中第%d个特征的信息增益为：%.3f" % (i, infoGain))

        # 更新最佳信息增益和最佳划分特征
        if infoGain >= bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i

    return bestFeature  # 返回最佳划分特征的索引

def C45_chooseBestFeatureToSplit(dataset):
    """
    使用C4.5算法选择最佳划分特征。

    Parameters:
    dataset (list): 数据集，每个元素是一个列表，表示一个样本的特征和标签（最后一个元素）。

    Returns:
    int: 最佳划分特征的索引。
    """
    numFeatures = len(dataset[0]) - 1  # 特征的数量（不包括标签）
    baseEnt = cal_entropy(dataset)  # 计算数据集的基础信息熵
    bestInfoGainRatio = 0.0  # 最佳信息增益率初始化
    bestFeature = -1  # 最佳划分特征初始化

    # 遍历所有特征
    for i in range(numFeatures):
        featList = [example[i] for example in dataset]  # 获取第i个特征的所有取值
        uniqueVals = set(featList)  # 将特征值列表转换为集合，得到唯一的特征取值

        newEnt = 0.0  # 初始化新的信息熵
        IV = 0.0  # 初始化信息增益率中的IV（Intrinsic Value）

        # 计算每种划分方式的信息熵和IV
        # TODO
        for value in uniqueVals:
            subdataset = splitdataset(dataset, i, value)
            prob = len(subdataset) / len(dataset)
            newEnt += prob * cal_entropy(subdataset)  # 计算信息熵
            IV += -prob * log(prob, 2) if prob != 0 else 0  # 计算IV

        infoGain = baseEnt - newEnt  # 计算信息增益
        # 避免除零错误
        if IV == 0:
            infoGainRatio = 0.0
        else:
            infoGainRatio = infoGain / IV  # 计算信息增益率

        print("C4.5中第%d个特征的信息增益率为：%.3f" % (i, infoGainRatio))

        # 更新最佳信息增益率和最佳划分特征
        if infoGainRatio >= bestInfoGainRatio:
            bestInfoGainRatio = infoGainRatio
            bestFeature = i

    return bestFeature  # 返回最佳划分特征的索引


def CART_chooseBestFeatureToSplit(dataset):
    """
    使用CART算法选择最佳划分特征。

    Parameters:
    dataset (list): 数据集，每个元素是一个列表，表示一个样本的特征和标签（最后一个元素）。

    Returns:
    int: 最佳划分特征的索引。
    """
    numFeatures = len(dataset[0]) - 1  # 特征的数量（不包括标签）
    bestGini = float('inf')  # 最佳基尼指数初始化为无穷大
    bestFeature = -1  # 最佳划分特征初始化

    # 遍历所有特征
    for i in range(numFeatures):
        featList = [example[i] for example in dataset]  # 获取第i个特征的所有取值
        uniqueVals = set(featList)  # 将特征值列表转换为集合，得到唯一的特征取值

        gini = 0.0  # 初始化基尼指数

        # 计算每种划分方式的基尼指数
        for value in uniqueVals:
            subdataset = splitdataset(dataset, i, value)  # 根据特征i和取值value划分子数据集
            classCounts = Counter([example[-1] for example in subdataset])  # 统计子数据集中每个类别的样本数
            # TODO
            prob = len(subdataset) / len(dataset)
            sub_gini = 1.0
            for count in classCounts.values():
                sub_prob = count / len(subdataset)
                sub_gini -= sub_prob ** 2
            gini += prob * sub_gini

        print(f"CART中第{i}个特征的基尼指数为：{gini:.3f}")

        # 更新最佳基尼指数和最佳划分特征
        if gini <= bestGini:
            bestGini = gini
            bestFeature = i

    return bestFeature  # 返回最佳划分特征的索引




def createTree(chooseBestFeatureToSplit, dataset, labels, test_dataset, label_dict, father_major, pre_pruning=False, post_pruning=False):
    """
    使用递归构建决策树。

    Parameters:
    chooseBestFeatureToSplit (function): 特征选择函数（如ID3、C4.5、CART算法中的选择函数）。
    dataset (list): 训练数据集。
    labels (list): 特征标签列表。
    test_dataset (list): 测试数据集。
    label_dict (dict): 特征-属性值字典。
    father_major (str): 父节点的多数标签（当某节点数据集为空时使用）。
    pre_pruning (bool): 是否进行预剪枝，默认为True。
    post_pruning (bool): 是否进行后剪枝，默认为True。

    Returns:
    dict: 构建好的决策树模型。
    """    
    classList = [example[-1] for example in dataset]

    # 当数据集为空时，返回父节点出现次数最多的分类标签
    if len(dataset) == 0:
        return father_major

    # 当数据集中所有样本的类别完全相同时，停止划分，返回类别标签
    if classList.count(classList[0]) == len(classList):
        return classList[0]

    # 当遍历完所有特征时，返回出现次数最多的类别标签
    if len(dataset[0]) == 1:
        return majorityCnt(classList)

    # 寻找最优索引
    # TODO
    bestFeat = chooseBestFeatureToSplit(dataset)
    bestFeatLabel = labels[bestFeat]

    print(u"此时最优索引为：" + (bestFeatLabel))

    DecisionTree = {bestFeatLabel: {}}
    featLabels = labels[:]
    # 得到节点所有的属性值
    uniqueVals = label_dict[labels[bestFeat]]
    del (labels[bestFeat])

    # 进行预剪枝
    if pre_pruning:
        # 计算划分前在测试集上的分类准确率
        ans = [example[-1] for example in test_dataset]
        result_counter = Counter([example[-1] for example in dataset])
        pre_split_output = result_counter.most_common(1)[0][0]
        pre_split_acc = cal_acc(test_output=[pre_split_output] * len(test_dataset), label=ans)

        # 计算划分后准确率
        outputs = []
        ans = []
        for value in sorted(uniqueVals):
            cut_testset = splitdataset(test_dataset, bestFeat, value)
            cut_dataset = splitdataset(dataset, bestFeat, value)

            for vec in cut_testset:
                ans.append(vec[-1])
            result_counter = Counter()
            if len(cut_dataset) == 0:
                leaf_output = majorityCnt(classList)
            else:
                # TODO
                leaf_output = majorityCnt([example[-1] for example in cut_dataset])

            outputs += [leaf_output] * len(cut_testset)
        post_split_acc = cal_acc(test_output=outputs, label=ans)

        # 如果划分后的准确率小于等于划分前的准确率，则禁止划分
        if post_split_acc <= pre_split_acc:
            return pre_split_output

    # 递归构建子树
    for value in sorted(uniqueVals):
        subLabels = labels[:]
        DecisionTree[bestFeatLabel][value] = createTree(
            chooseBestFeatureToSplit,
            splitdataset(dataset, bestFeat, value),
            subLabels,
            splitdataset(test_dataset, bestFeat, value),
            label_dict,
            majorityCnt(classList),
            pre_pruning,
            post_pruning)

    # 进行后剪枝
    if post_pruning and len(test_dataset) != 0:
        # 计算后剪枝前的准确率
        tree_output = classifytest(DecisionTree,
                                   featLabels,
                                   testDataSet=test_dataset)
        ans = [example[-1] for example in test_dataset]
        tree_acc = cal_acc(tree_output, ans)

        # 计算后剪枝后的准确率
        # TODO
        result_counter = Counter([example[-1] for example in dataset])
        post_prune_output = result_counter.most_common(1)[0][0]

        post_prune_acc = cal_acc([post_prune_output] * len(test_dataset), ans)

        # 如果剪枝后的准确率大于剪枝前的准确率，则进行剪枝
        if post_prune_acc > tree_acc:
            return post_prune_output

    return DecisionTree





if __name__ == '__main__':
    dataset_name = sys.argv[1]  # 从命令行参数获取数据集名称

    # 构建文件路径
    filename = os.path.join(dataset_name, 'train.txt')
    testfile = os.path.join(dataset_name, 'test.txt')
    labelfile = os.path.join(dataset_name, 'labels.txt')

    # 设置是否进行预剪枝和后剪枝
    # 可通过命令行参数控制：第3个参数为剪枝选项 (0=不剪枝, 1=预剪枝, 2=后剪枝)
    pruning_option = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    pre_pruning = (pruning_option == 1)
    post_pruning = (pruning_option == 2)

    # 读取训练集和标签
    dataset = read_dataset(filename)
    original_dataset = copy.deepcopy(dataset)  # 深拷贝原始数据集
    labels = read_labels(labelfile)

    # 构建标签-属性值字典
    label_dict = {}
    for i in range(len(labels)):
        label_dict[labels[i]] = set([example[i] for example in dataset])

    # 打印数据集的信息熵
    print("Ent(D):", cal_entropy(dataset))

    print(f'labels = {labels}')

    # 根据命令行参数选择使用的决策树算法
    dec_tree = sys.argv[2]
    if dec_tree == '1':
        chooseBestFeatureToSplit = ID3_chooseBestFeatureToSplit
        plotter = treePlotter.ID3_Tree  # 使用ID3算法时的可视化函数
        algo = "ID3"
    elif dec_tree == '2':
        chooseBestFeatureToSplit = C45_chooseBestFeatureToSplit
        plotter = treePlotter.C45_Tree  # 使用C4.5算法时的可视化函数
        algo = "C4.5"
    elif dec_tree == '3':
        chooseBestFeatureToSplit = CART_chooseBestFeatureToSplit
        plotter = treePlotter.CART_Tree  # 使用CART算法时的可视化函数
        algo = "CART"
    else:
        print("请输入正确的决策树算法编号：1.ID3 2.C4.5 3.CART")
        sys.exit(1)

    print(u"正在使用算法", algo)

    labels_tmp = labels[:]  # 拷贝标签列表，createTree会修改它
    # 构建决策树
    DecisionTree = createTree(chooseBestFeatureToSplit, dataset, labels_tmp,
                              test_dataset=read_dataset(testfile), label_dict=label_dict,
                              father_major=majorityCnt([example[-1] for example in dataset]),
                              pre_pruning=pre_pruning, post_pruning=post_pruning)
    
    # 打印决策树结构
    print(algo + " Decision Tree:\n", DecisionTree)
    
    # 可视化决策树
    plotter(DecisionTree)
    
    # 读取测试集数据
    testSet = read_dataset(testfile)
    
    # 获取测试集真实结果
    true_results = [vec[-1] for vec in testSet]
    print(algo + " Test Set True Results:\n", true_results)
    
    # 对测试集进行分类预测
    predicted_results = classifytest(DecisionTree, labels, testSet)
    print(algo + " Test Set Predicted Results:\n", predicted_results)
    
    # 计算分类准确率
    accuracy = cal_acc(predicted_results, true_results)
    print(algo + " Test Set Accuracy:\n", accuracy)