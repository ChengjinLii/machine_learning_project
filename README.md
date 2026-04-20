# Machine Learning Course Project

本仓库用于机器学习课程算法仿真与课程报告撰写，代码、实验结果和报告均位于当前目录下。

## 已实现内容

- `K-means` 聚类实验
- `SVM` 分类实验
- `BP` 神经网络实验（使用 `MLPClassifier` 进行仿真）
- `RBF` 神经网络实验
- 中文课程报告（LaTeX 源文件与 PDF）

## 目录结构

```text
machine_learning_project/
├── data/               # 自动下载的数据集（默认不上传 GitHub）
├── report/             # 中文 LaTeX 报告与 PDF
├── results/            # 实验图表与指标
├── scripts/            # 一键运行脚本
├── src/ml_course/      # 实验代码
└── requirements.txt    # Python 依赖
```

## 数据说明

`data/` 目录中的数据集文件不随仓库上传。运行实验脚本时，程序会自动下载所需开源数据集到本地 `data/raw/` 目录。

当前实验使用的数据集包括：

- Iris
- Wine
- Fashion-MNIST

## 运行方式

```bash
cd /data/chengjin/机器学习课程/machine_learning_project
python scripts/run_all.py
```

主脚本会自动：

1. 下载开源数据集到 `data/raw/`
2. 运行四类算法实验
3. 输出图表与指标到 `results/`

## 报告

- 报告源码：`report/main.tex`
- 报告 PDF：`report/main.pdf`

## 开源地址

`https://github.com/ChengjinLii/machine_learning_project`
