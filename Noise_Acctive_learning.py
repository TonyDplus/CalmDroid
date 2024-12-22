import random
import os
import argparse
import numpy as np
import joblib as jl
from skmultilearn.dataset import load_from_arff
from skmultilearn.ext import Meka
from sklearn.metrics import hamming_loss, f1_score, accuracy_score, zero_one_loss
from scipy.sparse import issparse

# 设置路径和分类器
path1 = r''
meka_classpath = r''
meka_classifiers = ['BR']
weka_classifiers = ['trees.RandomForest']

def get_label_acc(y_pred, y_true):
    y_pred = np.array(y_pred.todense())
    y_true = np.array(y_true.todense())
    right_count_list = np.array([0,0,0,0,0,0])
    for i in range(y_true.shape[0]):
        for j in range(y_true.shape[1]):
            if y_true[i][j] == y_pred[i][j]:
                right_count_list[j] += 1
    label_acc = right_count_list / y_true.shape[0]
    return label_acc


def get_acc2(y_pred, y_true, args):
    right_count = 0
    y_pred = np.array(y_pred.todense())
    y_true = np.array(y_true.todense())
    for i in range(y_true.shape[0]):
        flag = True
        flag2 = False
        for j in range(args.label_count):
            if y_true[i][j] == 0 and y_pred[i][j] == 1:
                flag = False
                break
        if flag:
            for j in range(args.label_count):
                if y_pred[i][j] == 1:
                    flag2 = True
                    break
        if flag and flag2:
            right_count += 1
    return format(right_count / y_true.shape[0], '.8f')


def evaluate_metrics(y_pred, y_true, args):
    # 如果 y_pred 或 y_true 是稀疏矩阵，转换为密集格式
    if issparse(y_pred):
        y_pred = y_pred.todense()
    if issparse(y_true):
        y_true = y_true.todense()

    # 确保 y_pred 和 y_true 是 NumPy 数组
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    # Hamming Loss
    hamming = hamming_loss(y_true, y_pred)

    # F1 Score (macro average)
    f1 = f1_score(y_true, y_pred, average='macro')

    # 0/1 Loss
    zero_one = zero_one_loss(y_true, y_pred)


    return hamming, f1, zero_one


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=10, type=int, help='迭代次数')
    parser.add_argument('--label_count', default=6, type=int, help='label的维度')
    parser.add_argument('--random_seed', default=42, type=int, help='随机种子')
    parser.add_argument('--batch_size', default=100, type=int, help='每批次添加的数据集大小')
    args = parser.parse_args()

    random.seed(args.random_seed)
    base_path = 'data'
    f_feature = os.path.join(base_path, 'final_feature.txt')
    label_num = args.label_count
    feature_num = 531  # 根据你的特征数量调整

    # 初始化MEKA
    meka = Meka(
        meka_classifier='meka.classifiers.multilabel.' + meka_classifiers[0],
        weka_classifier=weka_classifiers[0],
        meka_classpath=meka_classpath,
        java_command=path1
    )

    # 加载初始训练数据和测试数据
    # train_path = os.path.join(base_path, 'core_train_5noise_initial.arff')
    train_path = os.path.join(base_path, '180core_train_2.arff')
    test_path = os.path.join(base_path, '180test.arff')
    x_train, y_train = load_from_arff(train_path, label_count=label_num)
    x_test, y_test = load_from_arff(test_path, label_count=label_num)

    # print(x_train.shape)  # 应该显示 (n_samples, n_features)
    # print(y_train.shape)  # 应该显示 (n_samples, n_labels)

    # 训练初始模型
    model = meka.fit(X=x_train, y=y_train)
    y_pred = model.predict(x_test)
    initial_acc = get_acc2(y_pred, y_test, args)
    # print(f"Initial Accuracy: {initial_acc}")
    hamming, f1, zero_one= evaluate_metrics(y_pred, y_test, args)
    label_acc=get_label_acc(y_pred, y_test)

    print(f"Initial Accuracy: {initial_acc}")
    print(f"Initial Hamming Loss: {hamming}")
    print(f"Initial F1 Score: {f1}")
    print(f"Initial 0/1 Loss: {zero_one}")
    print(f"Initial Label Accuracy: {label_acc}")
    # 保存初始模型参数
    jl.dump(model, 'model_epoch_0.pkl')

    # 主动学习循环
    # for epoch in range(1, args.epoch + 1):
    for epoch in range(1, 10):
        # 加载新的训练数据
        # new_train_path = os.path.join(base_path, f'core_2714_epoch{epoch}_noisy_5.arff')
        new_train_path = os.path.join(base_path, f'180confuse_train.arff')
        x_new, y_new = load_from_arff(new_train_path, label_count=label_num)

        # 加载之前保存的模型
        model = jl.load(f'model_epoch_{epoch-1}.pkl')

        # 使用新数据继续训练
        model = meka.fit(X=x_new, y=y_new)  # 使用新数据继续训练

        # 保存训练后的模型
        jl.dump(model, f'model_epoch_{epoch}.pkl')

        # 在测试集上评估
        y_pred = model.predict(x_test)

        acc = get_acc2(y_pred, y_test, args)
        hamming, f1, zero_one = evaluate_metrics(y_pred, y_test, args)
        label_acc = get_label_acc(y_pred, y_test)
        print(f"Epoch {epoch}, Accuracy: {acc}")
        print(f"Epoch {epoch}, Hamming Loss: {hamming}")
        print(f"Epoch {epoch}, F1 Score: {f1}")
        print(f"Epoch {epoch}, 0/1 Loss: {zero_one}")
        print(f"Epoch {epoch}, Label Accuracy: {label_acc}")

if __name__ == "__main__":
    main()
