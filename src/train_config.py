# data and result direciton
data_dir      = "../Dataset"            # Contain train / validation / test / word vectors
result_dir    = "../results"            # Lab results

# training hyperparameters
max_len       = 60                     # max length for the sentence（block / padding）
batch_size    = 128
num_epochs    = 10
lr            = 2e-3
weight_decay  = 1e-5
embed_dim     = 50                      # same as wiki_word2vec_50.bin 
hidden_size   = 128
num_layers    = 1                       # RNN layers
dropout       = 0.3
kernel_sizes  = [3,4,5]                 # TextCNN convolution kernels
num_filters   = 100
bert_model_name = "bert-base-chinese"   # 也可换成 hfl/chinese-roberta-wwm-ext 等
bert_lr         = 2e-5                  # 通常比经典模型更小
bert_max_len    = 128                   # BERT 输入的最大分词长度
bert_batch_size = 16

# others
seed          = 2025
device        = "cuda"                  # "cpu" / "cuda"