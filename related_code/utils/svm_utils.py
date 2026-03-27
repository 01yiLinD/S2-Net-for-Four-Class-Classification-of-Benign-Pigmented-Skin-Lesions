"""
SVM工具模块 - 用于皮肤痣分类项目
提供特征提取、SVM训练和评估等功能
"""

import numpy as np
import torch
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os


def extract_features_for_svm(model, dataloader, device):
    """
    使用预训练模型提取特征供SVM使用[1](@ref)

    参数:
        model: 预训练模型
        dataloader: 数据加载器
        device: 计算设备

    返回:
        features: 提取的特征数组
        labels: 对应的标签数组
    """
    features = []
    labels = []

    model.eval()
    with torch.no_grad():
        for data, target in dataloader:
            data = data.to(device)

            # 获取模型特征（在全局平均池化层之前）
            # 注意：这里需要根据你的实际模型结构调整特征提取方式
            if hasattr(model, 'features'):
                # 如果模型有features属性（如自定义CNN）
                feature_output = model.features(data)
            elif hasattr(model, 'resnet'):
                # 如果使用ResNet模型
                x = model.resnet.conv1(data)
                x = model.resnet.bn1(x)
                x = model.resnet.relu(x)
                x = model.resnet.maxpool(x)
                x = model.resnet.layer1(x)
                x = model.resnet.layer2(x)
                x = model.resnet.layer3(x)
                feature_output = model.resnet.layer4(x)
            else:
                # 默认尝试获取卷积层输出
                feature_output = data
                for layer in list(model.children())[:-1]:  # 排除最后的全连接层
                    feature_output = layer(feature_output)

            # 展平特征
            feature_output = feature_output.view(feature_output.size(0), -1)
            feature_output = feature_output.cpu().numpy()

            features.extend(feature_output)
            labels.extend(target.numpy())

    return np.array(features), np.array(labels)


def create_svm_model(kernel='rbf', C=1.0, gamma='scale', probability=True):
    """
    创建SVM分类器管道[1,3](@ref)

    参数:
        kernel: 核函数类型 ('linear', 'poly', 'rbf', 'sigmoid')
        C: 正则化参数
        gamma: 核系数
        probability: 是否启用概率估计

    返回:
        svm_pipeline: SVM分类器管道
    """
    svm_pipeline = Pipeline([
        ('scaler', StandardScaler()),  # 特征标准化[1](@ref)
        ('svm', OneVsRestClassifier(SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            probability=probability,
            random_state=42
        )))
    ])

    return svm_pipeline


def train_svm_model(svm_model, train_features, train_labels):
    """
    训练SVM模型[1](@ref)

    参数:
        svm_model: SVM模型管道
        train_features: 训练特征
        train_labels: 训练标签

    返回:
        trained_model: 训练好的SVM模型
    """
    print("训练SVM模型...")
    svm_model.fit(train_features, train_labels)
    return svm_model


def evaluate_svm_model(svm_model, test_features, test_labels, class_names=None):
    """
    评估SVM模型性能[3](@ref)

    参数:
        svm_model: 训练好的SVM模型
        test_features: 测试特征
        test_labels: 测试标签
        class_names: 类别名称列表

    返回:
        accuracy: 准确率
        report: 分类报告
        cm: 混淆矩阵
    """
    # 预测
    test_preds = svm_model.predict(test_features)

    # 计算准确率
    accuracy = accuracy_score(test_labels, test_preds) * 100

    # 生成分类报告
    report = classification_report(test_labels, test_preds, target_names=class_names)

    # 生成混淆矩阵
    cm = confusion_matrix(test_labels, test_preds)

    return accuracy, report, cm


def save_svm_model(svm_model, file_path):
    """
    保存SVM模型到文件[1](@ref)

    参数:
        svm_model: 要保存的SVM模型
        file_path: 保存路径
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(svm_model, file_path)
    print(f"SVM模型已保存到: {file_path}")


def load_svm_model(file_path):
    """
    从文件加载SVM模型[1](@ref)

    参数:
        file_path: 模型文件路径

    返回:
        svm_model: 加载的SVM模型
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"模型文件不存在: {file_path}")

    svm_model = joblib.load(file_path)
    print(f"SVM模型已从 {file_path} 加载")
    return svm_model


def plot_confusion_matrix(cm, class_names, save_path=None):
    """
    绘制并保存混淆矩阵[3](@ref)

    参数:
        cm: 混淆矩阵
        class_names: 类别名称列表
        save_path: 保存路径（可选）
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"混淆矩阵已保存到: {save_path}")

    plt.show()


def perform_svm_training(feature_extractor, train_loader, test_loader, device,
                         output_dir='svm_results', class_names=None):
    """
    执行完整的SVM训练流程

    参数:
        feature_extractor: 特征提取模型
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        device: 计算设备
        output_dir: 输出目录
        class_names: 类别名称列表
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 提取特征
    print("提取训练特征...")
    train_features, train_labels = extract_features_for_svm(feature_extractor, train_loader, device)

    print("提取测试特征...")
    test_features, test_labels = extract_features_for_svm(feature_extractor, test_loader, device)

    print(f"训练特征形状: {train_features.shape}")
    print(f"测试特征形状: {test_features.shape}")

    # 创建和训练SVM模型
    svm_model = create_svm_model(kernel='rbf', C=10, gamma=0.01)
    trained_svm = train_svm_model(svm_model, train_features, train_labels)

    # 评估模型
    accuracy, report, cm = evaluate_svm_model(
        trained_svm, test_features, test_labels, class_names
    )

    print(f"SVM测试准确率: {accuracy:.2f}%")
    print("分类报告:")
    print(report)

    # 保存模型和结果
    model_path = os.path.join(output_dir, 'svm_model.pkl')
    save_svm_model(trained_svm, model_path)

    # 绘制和保存混淆矩阵
    if class_names:
        cm_path = os.path.join(output_dir, 'confusion_matrix.png')
        plot_confusion_matrix(cm, class_names, cm_path)

    # 保存评估结果
    report_path = os.path.join(output_dir, 'classification_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"SVM分类准确率: {accuracy:.2f}%\n\n")
        f.write("详细分类报告:\n")
        f.write(report)

    print(f"详细报告已保存到: {report_path}")

    return trained_svm, accuracy, report, cm