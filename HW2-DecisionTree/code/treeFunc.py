# coding:utf-8

from math import log
import operator
import treePlotter
from collections import Counter
import sys
import copy
import os



def read_labels(filename):
    """
    读取标签文件，返回标签列表。

    Parameters:
    filename (str): 标签文件的路径。

    Returns:
    list: 包含标签的列表。
    """
    with open(filename, 'r', encoding='utf-8') as f:
        line = f.readline().strip()  # 读取文件的第一行并去除首尾空白字符
    labels = line.split()  # 使用空格分割字符串，得到标签列表
    return labels

def read_dataset(filename):
    """
    读取数据集文件，返回数据集列表。

    Parameters:
    filename (str): 数据集文件的路径。

    Returns:
    list: 包含数据集的列表，每个元素是一个字符串列表（表示一行数据）。
    """
    with open(filename, 'r') as fr:
        all_lines = fr.readlines()  # 读取文件所有行到列表中
    dataset = []
    for line in all_lines:
        line = line.strip().split(',')  # 去除首尾空白字符，并按逗号分割字符串
        dataset.append(line)  # 将处理后的每行数据添加到数据集列表中
    return dataset




def splitdataset(dataset, axis, value):
    """
    按照给定特征划分数据集。

    Parameters:
    dataset (list): 数据集，每个元素是一个列表，表示一个样本的特征和标签（最后一个元素）。
    axis (int): 划分数据集的特征索引。
    value (any): 需要返回的特征的值。

    Returns:
    list: 满足特定特征值的子数据集。
    """
    retDataset = []  # 创建返回的数据集列表
    for featVec in dataset:  # 遍历数据集中的每个样本
        if featVec[axis] == value:  # 如果样本的特征值等于给定的值
            reducedFeatVec = featVec[:axis]  # 去掉给定索引的特征
            reducedFeatVec.extend(featVec[axis + 1:])  # 将符合条件的特征添加到返回的数据集列表
            retDataset.append(reducedFeatVec)  # 将处理后的样本添加到返回的数据集列表
    return retDataset



def majorityCnt(classList):
    """
    多数表决方法确定叶子节点的分类标签。

    Parameters:
    classList (list): 类标签列表，包含了所有样本在该叶子节点上的分类标签。

    Returns:
    str: 最终确定的分类标签。
    """
    classCount = {}  # 统计每个类别出现的次数
    for vote in classList:
        if vote not in classCount:
            classCount[vote] = 0
        classCount[vote] += 1

    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]  # 返回出现次数最多的类别作为该叶子节点的分类标签



def classify(inputTree, featLabels, testVec):
    """
    使用训练好的决策树进行分类。

    Parameters:
    inputTree (dict): 训练好的决策树模型。
    featLabels (list): 特征标签列表。
    testVec (list): 待分类的测试数据。

    Returns:
    str: 分类结果（类别标签）。
    """
    firstStr = list(inputTree.keys())[0]  # 获取决策树的根节点特征名称
    secondDict = inputTree[firstStr]  # 获取根节点下的子树信息
    featIndex = featLabels.index(firstStr)  # 获取根节点特征在特征标签列表中的索引
    classLabel = '0'  # 默认分类标签为'0'

    for key in secondDict.keys():
        if testVec[featIndex] == key:  # 如果测试数据的特征值与当前子树节点的值匹配
            if isinstance(secondDict[key], dict):  # 如果当前节点仍然是一个字典（非叶子节点）
                classLabel = classify(secondDict[key], featLabels, testVec)  # 递归调用分类函数
            else:  # 如果当前节点是叶子节点
                classLabel = secondDict[key]  # 获取叶子节点的分类标签

    return classLabel  # 返回分类结果（类别标签）


def classifytest(inputTree, featLabels, testDataSet):
    """
    对测试数据集进行分类。

    Parameters:
    inputTree (dict): 训练好的决策树模型。
    featLabels (list): 特征标签列表。
    testDataSet (list): 测试数据集，每个元素是一个待分类的样本。

    Returns:
    list: 包含所有样本分类结果（类别标签）的列表。
    """
    classLabelAll = []  # 存储所有样本的分类结果列表
    for testVec in testDataSet:
        classLabelAll.append(classify(inputTree, featLabels, testVec))  # 调用 classify 函数进行分类

    return classLabelAll  # 返回所有样本的分类结果列表


def cal_acc(test_output, label):
    """
    计算分类准确率。

    Parameters:
    test_output (list): 测试数据集分类结果。
    label (list): 测试数据集真实标签。

    Returns:
    float: 分类准确率。
    """
    assert len(test_output) == len(label)  # 断言确保测试数据集分类结果与真实标签长度一致
    count = 0  # 初始化计数器，记录分类正确的样本数量

    for index in range(len(test_output)):
        if test_output[index] == label[index]:  # 判断分类结果与真实标签是否一致
            count += 1  # 若一致，计数器加一

    accuracy = float(count / len(test_output))  # 计算分类准确率
    return accuracy  # 返回分类准确率


