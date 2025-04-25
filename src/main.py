import argparse, csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt

import train_config as C
import model as M
from utils import build_vocab, load_word2vec, make_loader, metric, make_loader_bert
from transformers import BertTokenizerFast
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_model(name, vocab_size, pretrained):
    if name == "cnn":
        return M.TextCNN(vocab_size, C.embed_dim, C.num_filters, C.kernel_sizes,
                         pretrained_emb=pretrained, dropout=C.dropout)
    if name == "lstm":
        return M.RNN_LSTM(vocab_size, C.embed_dim, C.hidden_size, C.num_layers,
                          C.dropout, pretrained=pretrained, bidirectional = True)
    if name == "gru":
        return M.RNN_GRU(vocab_size, C.embed_dim, C.hidden_size, C.num_layers,
                         C.dropout, pretrained=pretrained)
    if name == "mlp":
        return M.MLP(vocab_size, C.embed_dim, C.hidden_size, C.max_len, pretrained=pretrained,
                     dropout=C.dropout)
    if name == "bert":
        return M.BERTClassifier(C.bert_model_name, dropout=C.dropout)
    raise ValueError(f"unknown model {name}")

@torch.no_grad()
def evaluate(model_name, loader, net, device, use_amp=True):
    net.eval()
    tp = fp = fn = tn = 0
    loss_sum = 0
    for batch in loader:
        with autocast(enabled=use_amp):   
            if model_name == "bert":
                y       = batch["labels"].to(device)
                inputs  = {k: v.to(device) for k, v in batch.items() if k != "labels"}
                logits  = net(inputs)
            else:
                x, y    = batch
                x, y    = x.to(device), y.to(device)
                logits  = net(x)
            loss = F.cross_entropy(logits, y, reduction="sum")

        loss_sum += loss.item() 
        pred = logits.argmax(1)
        tp += ((pred == 1) & (y == 1)).sum().item()
        fp += ((pred == 1) & (y == 0)).sum().item()
        fn += ((pred == 0) & (y == 1)).sum().item()
        tn += ((pred == 0) & (y == 0)).sum().item()
    prec, recall, f1, acc = metric(tp, fp, fn, tn)
    return loss_sum / len(loader.dataset), acc, f1

# -------------------- Single model training --------------------
def run_one_model(model_name, loaders, vocab_size, emb_mat, device):
    train_loader, valid_loader, test_loader = loaders
    net = load_model(model_name, vocab_size, emb_mat).to(device)
    lr_used = C.bert_lr if model_name == "bert" else C.lr
    opt_cls = AdamW if model_name == "bert" else optim.Adam
    opt = opt_cls(net.parameters(), lr=lr_used, weight_decay=C.weight_decay)

    train_loss_hist, train_acc_hist = [], []
    val_loss_hist, val_acc_hist, val_f1_hist = [], [], []

    use_amp = (device.type == "cuda")
    scaler = GradScaler(enabled=use_amp)

    for epoch in range(1, C.num_epochs + 1):
        net.train()
        bar = tqdm(train_loader, desc=f"[{model_name}] Epoch {epoch}/{C.num_epochs}")
        total = correct = loss_sum = 0

        for batch in bar:
            opt.zero_grad()

            if model_name == "bert":
                y = batch["labels"].to(device)
                inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
                with autocast(enabled=use_amp):                           
                    logits = net(inputs)
                    loss = F.cross_entropy(logits, y)
            else:
                x, y = batch
                x, y = x.to(device), y.to(device)
                with autocast(enabled=use_amp):                            
                    logits = net(x)
                    loss = F.cross_entropy(logits, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            loss_sum += loss.item() * y.size(0)
            pred      = logits.argmax(1)
            total    += y.size(0)
            correct  += (pred == y).sum().item()

            bar.set_postfix(loss=loss_sum / total, acc=correct / total)

        # Record train
        train_loss_hist.append(loss_sum / total)
        train_acc_hist.append(correct / total)

        # Record val
        v_loss, v_acc, v_f1 = evaluate(model_name, valid_loader, net, device, use_amp=use_amp)
        val_loss_hist.append(v_loss)
        val_acc_hist.append(v_acc)
        val_f1_hist.append(v_f1)
        print(f"[{model_name}][Val] loss={v_loss:.4f} acc={v_acc:.4f} f1={v_f1:.4f}")

    # -------- Test --------
    test_loss, test_acc, test_f1 = evaluate(model_name, test_loader, net, device, use_amp=use_amp)
    print(f"[{model_name}][Test] loss={test_loss:.4f} acc={test_acc:.4f} f1={test_f1:.4f}")

    hist = dict(
        train_loss=train_loss_hist,
        train_acc=train_acc_hist,
        val_loss=val_loss_hist,
        val_acc=val_acc_hist,
        val_f1=val_f1_hist,
    )
    test_metrics = dict(loss=test_loss, acc=test_acc, f1=test_f1)
    return hist, test_metrics

# -------------------- Main entrance --------------------
def main(args):
    set_seed(C.seed)
    Path(C.result_dir).mkdir(parents=True, exist_ok=True)

    # -------- data preparation --------
    train_file = f"{C.data_dir}/train.txt"
    valid_file = f"{C.data_dir}/validation.txt"
    test_file  = f"{C.data_dir}/test.txt"
    w2v_bin    = f"{C.data_dir}/wiki_word2vec_50.bin"

    tokenizer = None
    if args.run_all or args.model == "bert":
        tokenizer = BertTokenizerFast.from_pretrained(C.bert_model_name)

    if args.run_all:
        model_list = ["cnn", "lstm", "gru", "mlp", "bert"]
    else:
        model_list = [args.model]

    # 传统模型的 vocab / word2vec 仅在它们被选中时才准备
    need_vocab = any(m in ["cnn", "lstm", "gru", "mlp"] for m in model_list)
    if need_vocab:
        word2id, _ = build_vocab([train_file, valid_file, test_file])
        emb_mat    = load_word2vec(w2v_bin, word2id, C.embed_dim)
    else:
        word2id, emb_mat = {}, None
    
    # 针对每个模型分别生成 DataLoader（防止混用）
    def get_loaders(model_name):
        if model_name == "bert":
            return (
                make_loader_bert(train_file, tokenizer, C.bert_max_len, C.bert_batch_size, True),
                make_loader_bert(valid_file, tokenizer, C.bert_max_len, C.bert_batch_size, False),
                make_loader_bert(test_file,  tokenizer, C.bert_max_len, C.bert_batch_size, False),
            )
        else:
            return (
                make_loader(train_file, word2id, C.max_len, C.batch_size, True),
                make_loader(valid_file, word2id, C.max_len, C.batch_size, False),
                make_loader(test_file,  word2id, C.max_len, C.batch_size, False),
            )
        
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("Using device:", device)
    if device.type == "cuda":
        print("CUDA available:", torch.cuda.is_available())
        print("CUDA device name:", torch.cuda.get_device_name(0))

    all_hist   = {}   
    all_tests  = {}   

    for m in model_list:
        loaders      = get_loaders(m)                 
        vocab_size   = len(word2id) if m != "bert" else 0
        hist, tests  = run_one_model(m, loaders, vocab_size, emb_mat, device)
        all_hist[m]  = hist
        all_tests[m] = tests

    # -------- Draw results --------
    epochs = range(1, C.num_epochs + 1)
    metric_names = ["train_loss", "train_acc", "val_loss", "val_acc", "val_f1"]
    titles = {
        "train_loss": "Train Loss",
        "train_acc":  "Train Accuracy",
        "val_loss":   "Validation Loss",
        "val_acc":    "Validation Accuracy",
        "val_f1":     "Validation F1",
    }

    if len(model_list) == 1:
        # single model
        mdl = model_list[0]
        hist = all_hist[mdl]

        # Loss 
        plt.figure()
        plt.plot(epochs, hist["train_loss"], label="Train")
        plt.plot(epochs, hist["val_loss"], label="Validation")
        plt.title(f"{mdl.upper()} Loss")
        plt.xlabel("Epoch")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{C.result_dir}/{mdl}_loss.png")
        plt.close()

        # Accuracy 
        plt.figure()
        plt.plot(epochs, hist["train_acc"], label="Train")
        plt.plot(epochs, hist["val_acc"], label="Validation")
        plt.title(f"{mdl.upper()} Accuracy")
        plt.xlabel("Epoch")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{C.result_dir}/{mdl}_acc.png")
        plt.close()

        # F1 
        plt.figure()
        plt.plot(epochs, hist["val_f1"], label="Validation F1")
        plt.title(f"{mdl.upper()} Validation F1")
        plt.xlabel("Epoch")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{C.result_dir}/{mdl}_f1.png")
        plt.close()

    else:
        # mutimodels
        for mname in metric_names:
            plt.figure()
            for mdl in model_list:
                plt.plot(epochs, all_hist[mdl][mname], label=mdl.upper())
            plt.title(titles[mname])
            plt.xlabel("Epoch")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{C.result_dir}/{mname}.png")
            plt.close()

    # -------- Draw test tabular --------
    print("\n=== Test Set Comparison ===")
    print("{:>6} | {:>8} {:>8} {:>8}".format("Model", "Loss", "Acc", "F1"))
    print("-"*38)
    with open(f"{C.result_dir}/test_scores.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "test_loss", "test_acc", "test_f1"])
        for mdl in model_list:
            s = all_tests[mdl]
            print("{:>6} | {:8.4f} {:8.4f} {:8.4f}".format(mdl.upper(), s["loss"], s["acc"], s["f1"]))
            writer.writerow([mdl, f"{s['loss']:.4f}", f"{s['acc']:.4f}", f"{s['f1']:.4f}"])
    print(f"\nAll figures and test_scores.csv have been saved to {C.result_dir}")

# -------------------- CLI --------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["cnn", "lstm", "gru", "mlp", "bert"], default="cnn",
                   help="the single one model to train")
    p.add_argument("--run_all", action="store_true",
                   help="run and compare all models")
    p.add_argument("--device", default=C.device)
    return p.parse_args()

if __name__ == "__main__":
    main(parse_args())