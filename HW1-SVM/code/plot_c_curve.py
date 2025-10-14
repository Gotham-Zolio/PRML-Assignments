import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from dataset import Dataset
import sys

# 配置：选择特征类型
# 可以通过命令行参数指定: python plot_c_curve.py [attrs|images]
# 默认使用原始图片
if len(sys.argv) > 1:
    feature_type = sys.argv[1].lower()
else:
    feature_type = 'images'  # 默认使用图片，可改为 'attrs' 使用属性

print(f"Feature type: {feature_type}")

# 构建数据集
print("Loading dataset...")
data = Dataset("../celeba")
X_imgs_train, X_attrs_train, Y_train = data.get_train_data()
X_imgs_test, X_attrs_test, Y_test = data.get_test_data()

# 根据配置选择特征
if feature_type == 'attrs':
    # 选择属性特征作为分类依据
    X_train = X_attrs_train
    X_test = X_attrs_test
    feature_name = "Attributes"
    print(f"Using attribute features, shape: {X_train.shape}")
elif feature_type == 'images':
    # 选择原始图片作为分类依据
    X_train = X_imgs_train.reshape(X_imgs_train.shape[0], -1)
    X_test = X_imgs_test.reshape(X_imgs_test.shape[0], -1)
    feature_name = "Raw Images"
    print(f"Using raw image features, shape: {X_train.shape}")
else:
    raise ValueError(f"Unknown feature type: {feature_type}. Use 'attrs' or 'images'.")

# 标准化训练集和测试集
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# 定义 C 值范围（根据特征类型选择不同范围）
if feature_type == 'attrs':
    C_values = np.logspace(-3, 2, 100)  # 从 0.001 到 100，共 100 个点
    print(f"Using C range for attributes: [{C_values[0]:.4e}, {C_values[-1]:.4e}]")
elif feature_type == 'images':
    # 原始图片：根据初步测试，最佳 C 值在较小范围内
    C_values = np.logspace(-6, 0, 100)  # 从 1e-6 到 1，共 100 个点
    print(f"Using C range for images: [{C_values[0]:.4e}, {C_values[-1]:.4e}]")
else:
    C_values = np.logspace(-6, 0, 100)
    print(f"Using default C range: [{C_values[0]:.4e}, {C_values[-1]:.4e}]")

print(f"Testing {len(C_values)} different C values...")

# 存储结果
accuracies = []
misclassified_counts = []

# 对每个 C 值训练 SVM 并评估
for i, C in enumerate(C_values):
    print(f"Progress: {i+1}/{len(C_values)}, C={C:.4e}")
    
    # 训练 SVM
    svm = SVC(kernel="linear", C=C)
    svm.fit(X_train_std, Y_train)
    
    # 预测
    Y_pred = svm.predict(X_test_std)
    
    # 计算指标
    misclassified = (Y_test != Y_pred).sum()
    accuracy = svm.score(X_test_std, Y_test)
    
    misclassified_counts.append(misclassified)
    accuracies.append(accuracy)
    
    print(f"  Misclassified: {misclassified}, Accuracy: {accuracy:.4f}")

# 创建图形
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 绘制 Misclassified samples vs C
ax1.semilogx(C_values, misclassified_counts, 'b-o', linewidth=2, markersize=2)
ax1.set_xlabel('C (Penalty Parameter)', fontsize=12)
ax1.set_ylabel('Misclassified Samples', fontsize=12)
ax1.set_title(f'Misclassified Samples vs C Value ({feature_name})', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
best_c_mis = C_values[np.argmin(misclassified_counts)]
ax1.axvline(x=best_c_mis, color='r', linestyle='--', alpha=0.5, label=f'Optimal C = {best_c_mis:.4e}')
ax1.legend()

# 绘制 Accuracy vs C
ax2.semilogx(C_values, accuracies, 'g-o', linewidth=2, markersize=2)
ax2.set_xlabel('C (Penalty Parameter)', fontsize=12)
ax2.set_ylabel('Accuracy', fontsize=12)
ax2.set_title(f'Accuracy vs C Value ({feature_name})', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
best_c_acc = C_values[np.argmax(accuracies)]
ax2.axvline(x=best_c_acc, color='r', linestyle='--', alpha=0.5, label=f'Optimal C = {best_c_acc:.4e}')
ax2.legend()

plt.tight_layout()

# 创建 log 文件夹（如果不存在）
import os
log_dir = '../log'
os.makedirs(log_dir, exist_ok=True)

# 保存图片（根据特征类型命名）
output_path = f'{log_dir}/c_value_curves_{feature_type}.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n图片已保存到: {output_path}")

# 显示图片
plt.show()

# 输出最佳结果
best_idx = np.argmax(accuracies)
print("\n" + "="*50)
print("最佳结果:")
print(f"特征类型: {feature_name}")
print(f"最佳 C 值: {C_values[best_idx]:.4e}")
print(f"最佳准确率: {accuracies[best_idx]:.4f}")
print(f"最少错分样本数: {misclassified_counts[best_idx]}")
print("="*50)

# 保存详细结果到文本文件（根据特征类型命名）
result_file = f'{log_dir}/c_value_results_{feature_type}.txt'
with open(result_file, 'w', encoding='utf-8') as f:
    f.write(f"Impact of C Value on SVM Performance ({feature_name})\n")
    f.write("="*70 + "\n\n")
    f.write(f"Feature Type: {feature_name}\n")
    f.write(f"Training samples: {len(Y_train)}, Test samples: {len(Y_test)}\n")
    f.write(f"Feature dimension: {X_train.shape[1]}\n\n")
    f.write(f"{'C Value':<18} {'Accuracy':<12} {'Misclassified':<15}\n")
    f.write("-"*70 + "\n")
    for c, acc, mis in zip(C_values, accuracies, misclassified_counts):
        f.write(f"{c:<18.6e} {acc:<12.4f} {mis:<15d}\n")
    f.write("\n" + "="*70 + "\n")
    f.write(f"Best C Value: {C_values[best_idx]:.6e}\n")
    f.write(f"Best Accuracy: {accuracies[best_idx]:.4f}\n")
    f.write(f"Minimum Misclassified: {misclassified_counts[best_idx]}\n")

print(f"详细结果已保存到: {result_file}")
