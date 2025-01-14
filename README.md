# Code for 'Dynamic MLP for Fine-Grained Image Classification by Leveraging Geographical and Temporal Information'

<p align="center"> <img src="figs/structure.svg" width="100%"></p>
<a href="https://arxiv.org/pdf/2203.03253.pdf">Dynamic MLP</a>, which is parameterized by the learned embeddings of variable locations and dates to help fine-grained image classification.


- **Local test**

| Backbone | Size | 90epoch Acc@1 | 110epoch Acc@1 | 130epoch Acc@1 | log |
| -------------- | :---: | :--------: | :-------: | :------------: | :--------: | 
| ResNet-50      |  224  |  .. | .. |.. |    [log](logs/log_inat21-mini_90epoch_r50_image-only_67.924_top1_acc.txt)     |
| + Dynamic MLP_c  |  224  | **77.905** |  **78.373** |  **78.432**   |  [log](outputs/logs/log_inat21-mini_90epoch_r50_dynamic-mlp-c_78.751_top1_acc.txt)   |
| + Dynamic MLP_d  |  224  | **77.581** | **77.581** | **77.999**(120epoch) | [log](logs/log_inat21-mini_90epoch_r50_dynamic-mlp-c_78.751_top1_acc.txt)   |




## Repository Setup
1. Create a fresh conda environment, and install all dependencies.
    ```
    conda create -n env_DynamicMLP python=3.6
    conda activate env_DynamicMLP
    git clone https://github.com/xxayt/DynamicMLP.git
    cd DynamicMLP
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
    ```

2. Install pytorch
    ```
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch==1.7.1 torchvision==0.8.2
    ```

### Pretrained Models

Get pretrained models for SK-Res2Net following [here](checkpoints/README.md).

### Data Preparation

Get datasets following [here](datasets/README.md).

## Train the model
### 1. Train image-only model
Specify ```--image_only``` for training image-only models.
- ResNet-50 (67.924% Top-1 acc)
```python
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 train.py \
  --name res50_image_only \
  --data 'inat21_mini' \
  --data_dir 'path/to/your/data' \
  --model_file 'resnet' \
  --model_name 'resnet50' \
  --pretrained \
  --batch_size 512 \
  --start_lr 0.04 \
  --image_only
```

- SK-Res2Net-101 (76.102% Top-1 acc)
```python
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 train.py \
  --name sk2_image_only \
  --data 'inat21_mini' \
  --data_dir 'path/to/your/data' \
  --model_file 'sk2res2net' \
  --model_name 'sk2res2net101' \
  --pretrained \
  --batch_size 512 \
  --start_lr 0.04 \
  --image_only
```

### 2. Train dynamic MLP model
- ResNet-50 (78.751% Top-1 acc)
```python
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 train.py \
  --name res50_dynamic_mlp \
  --data 'inat21_mini' \
  --data_dir 'path/to/your/data' \
  --model_file 'resnet_dynamic_mlp' \
  --model_name 'resnet50' \
  --pretrained \
  --batch_size 512 \
  --start_lr 0.04
```

- SK-Res2Net-101 (84.694% Top-1 acc)
```python
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 train.py \
  --name sk2_dynamic_mlp \
  --data 'inat21_mini' \
  --data_dir 'path/to/your/data' \
  --model_file 'sk2res2net_dynamic_mlp' \
  --model_name 'sk2res2net101' \
  --pretrained \
  --batch_size 512 \
  --start_lr 0.04
```

## Test the model
Specify ```--resume``` and ```--evaluate``` for inference and ```--image_only``` for testing image-only models.
```python
python3 train.py \
  --name sk2_dynamic_mlp \
  --data 'inat21_mini' \
  --data_dir 'path/to/your/data' \
  --model_file 'sk2res2net_dynamic_mlp' \
  --model_name 'sk2res2net101' \
  --resume 'path/to/your/checkpoint' \
  --evaluate
```

## Model Zoo
### iNaturalist 2021 mini (90 epoch)

| Backbone       | Size  |   Acc@1    |                                      Log                                      |                                                                     Download                                                                     |
| -------------- | :---: | :--------: | :---------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------: |
| ResNet-50      |  224  |   67.924   |    [log](logs/log_inat21-mini_90epoch_r50_image-only_67.924_top1_acc.txt)     |    [model](https://github.com/ylingfeng/DynamicMLP/releases/download/v0.0/checkpoint_inat21-mini_90epoch_r50_image-only_67.924_top1_acc.pth)     |
| + Dynamic MLP  |  224  | **78.751** |   [log](logs/log_inat21-mini_90epoch_r50_dynamic-mlp-c_78.751_top1_acc.txt)   |   [model](https://github.com/ylingfeng/DynamicMLP/releases/download/v0.0/checkpoint_inat21-mini_90epoch_r50_dynamic-mlp-c_78.751_top1_acc.pth)   |
| SK-Res2Net-101 |  224  |   76.102   |  [log](logs/log_inat21-mini_90epoch_sk2-101_image-only_76.102_top1_acc.txt)   |  [model](https://github.com/ylingfeng/DynamicMLP/releases/download/v0.0/checkpoint_inat21-mini_90epoch_sk2-101_image-only_76.102_top1_acc.pth)   |
| + Dynamic MLP  |  224  | **84.694** | [log](logs/log_inat21-mini_90epoch_sk2-101_dynamic-mlp-c_84.694_top1_acc.txt) | [model](https://github.com/ylingfeng/DynamicMLP/releases/download/v0.0/checkpoint_inat21-mini_90epoch_sk2-101_dynamic-mlp-c_84.694_top1_acc.pth) |

