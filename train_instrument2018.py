"""
使用 python/ 目录下的 SASRec 在 Instrument2018 数据集上训练，并提取 item embedding
"""
import os
import sys
import json
import argparse
import numpy as np
import torch

# 添加 python 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

from model import SASRec
from utils import data_partition, WarpSampler, evaluate, evaluate_valid
import time


def prepare_data_from_inter(inter_file, output_file):
    """
    从 .inter 文件准备数据，转换为 user_id item_id 格式
    """
    print(f"正在从 {inter_file} 准备数据...")
    
    user_item_pairs = []
    user_map = {}  # 原始 user_id -> 数字 ID
    item_map = {}  # 原始 item_id -> 数字 ID
    user_counter = 1
    item_counter = 1
    
    with open(inter_file, 'r') as f:
        # 跳过 header
        header = f.readline()
        
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 4:
                continue
            
            user_id_raw = parts[0]
            item_id_raw = parts[1]
            timestamp = float(parts[3])
            
            # 映射 user
            if user_id_raw not in user_map:
                user_map[user_id_raw] = user_counter
                user_counter += 1
            
            # 映射 item
            if item_id_raw not in item_map:
                item_map[item_id_raw] = item_counter
                item_counter += 1
            
            user_item_pairs.append((
                user_map[user_id_raw],
                item_map[item_id_raw],
                timestamp
            ))
    
    # 按 timestamp 排序
    user_item_pairs.sort(key=lambda x: x[2])
    
    # 写入文件
    with open(output_file, 'w') as f:
        for user_id, item_id, _ in user_item_pairs:
            f.write(f"{user_id} {item_id}\n")
    
    print(f"数据准备完成: {len(user_map)} 个用户, {len(item_map)} 个物品")
    print(f"输出文件: {output_file}")
    
    # 保存映射关系
    map_file = output_file.replace('.txt', '_map.json')
    with open(map_file, 'w') as f:
        json.dump({
            'user_map': user_map,
            'item_map': item_map,
            'reverse_item_map': {v: k for k, v in item_map.items()}
        }, f, indent=2)
    print(f"映射关系保存到: {map_file}")
    
    return user_map, item_map


def extract_item_embeddings(model, itemnum, output_file):
    """
    从训练好的模型中提取 item embedding
    """
    print(f"\n正在提取 item embedding...")
    
    # 获取 item embedding 权重 (shape: [itemnum+1, hidden_units])
    item_embeddings = model.item_emb.weight.data.cpu().numpy()
    
    # 去掉 padding (index 0)
    item_embeddings = item_embeddings[1:, :]  # shape: [itemnum, hidden_units]
    
    print(f"Item embedding shape: {item_embeddings.shape}")
    print(f"保存到: {output_file}")
    
    np.save(output_file, item_embeddings)
    
    return item_embeddings


def main():
    parser = argparse.ArgumentParser()
    
    # 数据相关
    parser.add_argument('--inter_file', type=str, 
                        default='RecBole/dataset/Instrument2018/Instrument2018.inter',
                        help='.inter 文件路径')
    parser.add_argument('--dataset', type=str, default='Instrument2018',
                        help='数据集名称')
    parser.add_argument('--prepare_data', action='store_true',
                        help='是否重新准备数据')
    
    # 训练相关
    parser.add_argument('--train_dir', type=str, default='default',
                        help='训练输出目录后缀')
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--maxlen', type=int, default=50,
                        help='最大序列长度')
    parser.add_argument('--hidden_units', type=int, default=256,
                        help='embedding 维度')
    parser.add_argument('--num_blocks', type=int, default=2,
                        help='Transformer block 数量')
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--num_heads', type=int, default=2,
                        help='multi-head attention 的 head 数量')
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--l2_emb', type=float, default=0.0,
                        help='embedding 的 L2 正则化系数')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--norm_first', action='store_true', default=False,
                        help='是否在 attention 前做 layer norm')
    
    # 推理相关
    parser.add_argument('--inference_only', action='store_true',
                        help='仅推理模式')
    parser.add_argument('--state_dict_path', type=str, default=None,
                        help='加载模型权重路径')
    parser.add_argument('--extract_emb_only', action='store_true',
                        help='仅提取 embedding')
    
    args = parser.parse_args()
    
    # 保存原始工作目录
    original_dir = os.getcwd()
    python_dir = os.path.join(original_dir, 'python')
    
    # 准备数据
    data_file = os.path.join(python_dir, 'data', f'{args.dataset}.txt')
    if args.prepare_data or not os.path.exists(data_file):
        os.makedirs(os.path.join(python_dir, 'data'), exist_ok=True)
        inter_file_abs = os.path.join(original_dir, args.inter_file)
        prepare_data_from_inter(inter_file_abs, data_file)
    
    # 如果只是提取 embedding
    if args.extract_emb_only:
        if args.state_dict_path is None:
            print("错误: 提取 embedding 需要指定 --state_dict_path")
            return
        
        # 切换到 python 目录以加载数据
        os.chdir(python_dir)
        dataset = data_partition(args.dataset)
        [user_train, user_valid, user_test, usernum, itemnum] = dataset
        os.chdir(original_dir)
        
        # 创建模型并加载权重
        model = SASRec(usernum, itemnum, args).to(args.device)
        state_dict_path = os.path.join(original_dir, args.state_dict_path)
        model.load_state_dict(torch.load(state_dict_path, map_location=torch.device(args.device)))
        
        # 提取 embedding
        output_dir = os.path.join(original_dir, args.dataset + '_' + args.train_dir)
        emb_file = os.path.join(output_dir, f'{args.dataset}_item_emb.npy')
        extract_item_embeddings(model, itemnum, emb_file)
        return
    
    # 创建输出目录
    output_dir = os.path.join(original_dir, args.dataset + '_' + args.train_dir)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    # 保存参数
    with open(os.path.join(output_dir, 'args.txt'), 'w') as f:
        f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
    
    # 切换到 python 目录以加载数据
    os.chdir(python_dir)
    dataset = data_partition(args.dataset)
    os.chdir(original_dir)
    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    
    num_batch = (len(user_train) - 1) // args.batch_size + 1
    cc = sum(len(user_train[u]) for u in user_train)
    print(f'平均序列长度: {cc / len(user_train):.2f}')
    print(f'用户数: {usernum}, 物品数: {itemnum}')
    
    # 创建日志文件
    log_file = open(os.path.join(output_dir, 'log.txt'), 'w')
    log_file.write('epoch (val_ndcg, val_hr) (test_ndcg, test_hr)\n')
    
    # 创建 sampler 和模型
    sampler = WarpSampler(user_train, usernum, itemnum, 
                          batch_size=args.batch_size, 
                          maxlen=args.maxlen, 
                          n_workers=3)
    model = SASRec(usernum, itemnum, args).to(args.device)
    
    # 初始化参数
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass
    
    model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0
    
    model.train()
    
    # 加载预训练权重
    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            state_dict_path = os.path.join(original_dir, args.state_dict_path)
            model.load_state_dict(torch.load(state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
            print(f'从 epoch {epoch_start_idx} 继续训练')
        except Exception as e:
            print(f'加载权重失败: {e}')
    
    # 推理模式
    if args.inference_only:
        model.eval()
        t_test = evaluate(model, dataset, args)
        print(f'Test (NDCG@10: {t_test[0]:.4f}, HR@10: {t_test[1]:.4f})')
        sampler.close()
        log_file.close()
        return
    
    # 训练
    bce_criterion = torch.nn.BCEWithLogitsLoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    
    best_val_ndcg, best_val_hr = 0.0, 0.0
    best_test_ndcg, best_test_hr = 0.0, 0.0
    T = 0.0
    t0 = time.time()
    
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        for step in range(num_batch):
            u, seq, pos, neg = sampler.next_batch()
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            
            pos_logits, neg_logits = model(u, seq, pos, neg)
            pos_labels = torch.ones(pos_logits.shape, device=args.device)
            neg_labels = torch.zeros(neg_logits.shape, device=args.device)
            
            adam_optimizer.zero_grad()
            indices = np.where(pos != 0)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            
            # L2 正则化
            for param in model.item_emb.parameters():
                loss += args.l2_emb * torch.sum(param ** 2)
            
            loss.backward()
            adam_optimizer.step()
            
            if step % 100 == 0:
                print(f"Epoch {epoch} Step {step}/{num_batch}: loss={loss.item():.4f}")
        
        # 每 20 个 epoch 评估一次
        if epoch % 20 == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('评估中...', end='')
            
            t_test = evaluate(model, dataset, args)
            t_valid = evaluate_valid(model, dataset, args)
            
            print(f'\nEpoch {epoch}, 时间: {T:.1f}s')
            print(f'Valid (NDCG@10: {t_valid[0]:.4f}, HR@10: {t_valid[1]:.4f})')
            print(f'Test (NDCG@10: {t_test[0]:.4f}, HR@10: {t_test[1]:.4f})')
            
            # 保存最佳模型
            if t_valid[0] > best_val_ndcg or t_valid[1] > best_val_hr or \
               t_test[0] > best_test_ndcg or t_test[1] > best_test_hr:
                best_val_ndcg = max(t_valid[0], best_val_ndcg)
                best_val_hr = max(t_valid[1], best_val_hr)
                best_test_ndcg = max(t_test[0], best_test_ndcg)
                best_test_hr = max(t_test[1], best_test_hr)
                
                fname = f'SASRec.epoch={epoch}.lr={args.lr}.layer={args.num_blocks}.head={args.num_heads}.hidden={args.hidden_units}.maxlen={args.maxlen}.pth'
                model_path = os.path.join(output_dir, fname)
                torch.save(model.state_dict(), model_path)
                print(f'保存模型: {model_path}')
            
            log_file.write(f'{epoch} {t_valid} {t_test}\n')
            log_file.flush()
            t0 = time.time()
            model.train()
    
    # 训练结束，保存最终模型
    fname = f'SASRec.epoch={args.num_epochs}.lr={args.lr}.layer={args.num_blocks}.head={args.num_heads}.hidden={args.hidden_units}.maxlen={args.maxlen}.pth'
    final_model_path = os.path.join(output_dir, fname)
    torch.save(model.state_dict(), final_model_path)
    print(f'\n训练完成! 最终模型保存到: {final_model_path}')
    
    # 提取 item embedding
    emb_file = os.path.join(output_dir, f'{args.dataset}_item_emb.npy')
    extract_item_embeddings(model, itemnum, emb_file)
    
    log_file.close()
    sampler.close()
    print("完成!")


if __name__ == '__main__':
    main()
