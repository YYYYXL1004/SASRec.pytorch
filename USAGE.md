# 使用说明

## 快速开始

```bash
# 训练模型并提取 item embedding
bash run_instrument2018.sh train

# 或者直接使用 Python
python train_instrument2018.py --prepare_data --dataset Instrument2018
```

## 主要参数 (与 RecBole 版本对齐)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--hidden_units` | 256 | Embedding 维度 |
| `--maxlen` | 50 | 最大序列长度 |
| `--num_blocks` | 2 | Transformer 层数 |
| `--num_heads` | 2 | Attention head 数量 |
| `--batch_size` | 2048 | Batch size |
| `--dropout_rate` | 0.5 | Dropout 比率 |
| `--num_epochs` | 200 | 训练轮数 |

## 输出文件

- **模型权重**: `Instrument2018_default/SASRec.*.pth`
- **Item embedding**: `Instrument2018_default/Instrument2018_item_emb.npy`
- **ID 映射**: `python/data/Instrument2018_map.json`

## 使用 Item Embedding

```python
import numpy as np
import json

# 加载 embedding
item_emb = np.load('Instrument2018_default/Instrument2018_item_emb.npy')

# 加载 ID 映射
with open('python/data/Instrument2018_map.json', 'r') as f:
    mapping = json.load(f)
    item_map = mapping['item_map']  # 原始 ID -> 数字 ID

# 获取某个物品的 embedding
original_item_id = 'B000VSM4MS'
numeric_id = item_map[original_item_id]
embedding = item_emb[numeric_id - 1]  # 注意: 数字 ID 从 1 开始
```
