# CycleGAN-PyTorch

一个基于 **PyTorch Lightning** 与 **Hydra** 的 CycleGAN 实现，旨在提供清晰、灵活且易于扩展的深度学习训练框架。项目继承了 Lightning-Hydra-Template 的诸多优点，并对 CycleGAN 进行了简洁实现。

## 特性一览

- **模块化配置**：所有训练参数均以 Hydra 配置形式管理，便于复现和实验对比。
- **易于扩展**：`src` 目录下提供了数据、模型与训练脚本的完整结构，可快速添加新的组件。
- **友好的训练流程**：支持 GPU/CPU 训练，内置日志、回调和学习率调度等常用功能。

## 快速开始

1. 安装依赖

   ```bash
   pip install -r requirements.txt
   ```

   或使用 `environment.yaml` 创建 Conda 环境。

2. 下载数据集（以 `summer2winter_yosemite` 为例）

   ```bash
   bash scripts/download.sh summer2winter_yosemite
   ```

3. 运行训练

   ```bash
   python src/train.py data=cycle-gan model=cycle-gan trainer=gpu
   ```

   训练日志和模型权重将保存在 `logs/` 目录下。

## 目录结构

- `configs/`：Hydra 配置文件
- `src/`：核心代码（数据集、模型和训练脚本等）
- `scripts/`：辅助脚本，如数据下载
- `tests/`：单元测试

## 许可证

本项目基于 MIT License 开源，欢迎自由使用与贡献。
