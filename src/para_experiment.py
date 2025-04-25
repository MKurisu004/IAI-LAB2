# LAB2/src/para_experiment.py
"""
批量调参实验脚本
示例:
    python para_experiment.py --param batch_size
    python para_experiment.py --param lr
"""

import argparse, importlib, time, copy
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch

import train_config as C
import test_para as T          # 参数网格
import main as runner          # 我们直接重用 main.py 里的工具函数
from utils import build_vocab, load_word2vec, make_loader
from utils import make_loader_bert      # bert 不测，但函数必须 import
from transformers import BertTokenizerFast

# ---------- 统一的实验函数 ----------
def run_single(model_name, cfg, device, want_time=False):
    """
    cfg: 当前遍历到的一份 train_config 副本
    返回 (loss, f1, seconds)
    """
    # === 重新构建数据管道 ===
    train_file = f"{cfg.data_dir}/train.txt"
    valid_file = f"{cfg.data_dir}/validation.txt"
    test_file  = f"{cfg.data_dir}/test.txt"
    w2v_bin    = f"{cfg.data_dir}/wiki_word2vec_50.bin"

    word2id, _ = build_vocab([train_file, valid_file, test_file])
    emb_mat    = load_word2vec(w2v_bin, word2id, cfg.embed_dim)

    loaders = (
        make_loader(train_file, word2id, cfg.max_len, cfg.batch_size, True),
        make_loader(valid_file, word2id, cfg.max_len, cfg.batch_size, False),
        make_loader(test_file,  word2id, cfg.max_len, cfg.batch_size, False),
    )

    start = time.time()
    hist, test_metrics = runner.run_one_model(
        model_name, loaders, len(word2id), emb_mat, device
    )
    secs = time.time() - start

    # --- 用 test_metrics 取值 -----------------
    loss = test_metrics["loss"]
    f1   = test_metrics["f1"]
    return (loss, f1, secs if want_time else None)

# ---------- 画图 ----------
def plot_lines(x, ys, labels, title, save_path, xlabel):
    plt.figure()
    for y, lab in zip(ys, labels):
        plt.plot(x, y, marker="o", label=lab)
    plt.title(title); plt.xlabel(xlabel); plt.legend(); plt.tight_layout()
    plt.savefig(save_path); plt.close()

# ---------- 主控 ----------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--param", choices=list(T.GRID.keys()), required=True,
                   help="要实验的超参数名")
    p.add_argument("--device", default=C.device)
    args = p.parse_args()

    param = args.param
    values = T.GRID[param]
    Path(C.result_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Running param search for {param} on {device}")

    # 存结果
    results = {}

    # baseline config 备份
    PARAM_KEYS = ["batch_size", "lr", "max_len", "dropout", "kernel_sizes"]
    base_cfg = {k: getattr(C, k) for k in PARAM_KEYS}

    for v in tqdm(values, desc=f"{param} grid"):
        print(f"\n=== {param} = {v} ===")
        # 把属性写入 train_config
        setattr(C, param if param != "kernel_size" else "kernel_sizes", v)
        if param == "lr":         # 学习率只影响 cnn，所以直接改 C.lr
            C.lr = v
        if param == "dropout":
            C.dropout = v

        if param == "batch_size":          # 只测 cnn
            loss, f1, secs = run_single("cnn", C, device, want_time=True)
            results.setdefault("loss", []).append(loss)
            results.setdefault("secs", []).append(secs)
        elif param == "lr":
            loss, f1, _ = run_single("cnn", C, device)
            results.setdefault("loss", []).append(loss)
            results.setdefault("f1", []).append(f1)
        elif param == "kernel_size":
            loss, f1, _ = run_single("cnn", C, device)
            results.setdefault("loss", []).append(loss)
            results.setdefault("f1", []).append(f1)
        elif param in ["max_len", "dropout"]:
            # gru + lstm
            for mdl in ["gru", "lstm"]:
                loss, f1, _ = run_single(mdl, C, device)
                results.setdefault(f"{mdl}_loss", []).append(loss)
                results.setdefault(f"{mdl}_f1", []).append(f1)
        else:
            raise ValueError(param)

    # ---------- 出图 ----------
    x_axis = values
    if param == "batch_size":
        plot_lines(x_axis, [results["secs"]], ["CNN"], "Train Time vs Batch", 
                   f"{C.result_dir}/batch_time.png", "Batch Size")
        plot_lines(x_axis, [results["loss"]], ["CNN"], "Val Loss vs Batch", 
                   f"{C.result_dir}/batch_loss.png", "Batch Size")
    elif param == "lr":
        plot_lines(x_axis, [results["loss"]], ["CNN"], "Val Loss vs LR", 
                   f"{C.result_dir}/lr_loss.png", "LR")
        plot_lines(x_axis, [results["f1"]], ["CNN"], "Val F1 vs LR", 
                   f"{C.result_dir}/lr_f1.png", "LR")
    elif param == "kernel_size":
        ks_label = [str(k) for k in values]
        plot_lines(ks_label, [results["loss"]], ["CNN"], "Val Loss vs KernelSizes",
                   f"{C.result_dir}/ks_loss.png", "kernel_sizes list")
        plot_lines(ks_label, [results["f1"]], ["CNN"], "Val F1 vs KernelSizes",
                   f"{C.result_dir}/ks_f1.png", "kernel_sizes list")
    elif param in ["max_len", "dropout"]:
        tag = "maxlen" if param == "max_len" else "dropout"
        plot_lines(x_axis, [results["gru_loss"], results["lstm_loss"]],
                   ["GRU", "LSTM"], f"Val Loss vs {param}", 
                   f"{C.result_dir}/{tag}_loss.png", param)
        plot_lines(x_axis, [results["gru_f1"], results["lstm_f1"]],
                   ["GRU", "LSTM"], f"Val F1 vs {param}",
                   f"{C.result_dir}/{tag}_f1.png", param)

    # 恢复原始配置
    for k, v in base_cfg.items():
        setattr(C, k, v)

    print("Experiment finished. Figures saved to", C.result_dir)

if __name__ == "__main__":
    main()
