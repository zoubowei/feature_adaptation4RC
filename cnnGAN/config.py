import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
CNN = True
POS = True
NER = True
DIST = True

kernel_sizes1 = [3]
kernel_sizes2 = [3]
kernel_num = 100
dropout = 0.5
hidden_size = 200
BATCH = 100
EPOCH = 5000
diff_rate =  1.0
PRINT = 200
epochs = 100


M = 1.0
C = 0.5
words_dim = 50
pos_dim = 30
ner_dim = 30
dist_dim = 20

relation_path = "data/relations.txt"
EN_vocab_file = "model/en.vocab"
EN_emb_file = "model/en.embnpy"
CH_vocab_file = "model/ch.vocab"
CH_emb_file = "model/ch.embnpy"
'''
EN_emb_file = 'model/vector.ch.txt.example'
CH_emb_file = 'model/vector.ch.txt.example'
'''
en_max_len = 100
ch_max_len = 100
en_max_dist = 30
ch_max_dist = 30

#position
en_dist = 40
ch_dist = 40

#file
EN_train_file = "data/ace05_en.raw.choice"
EN_chtrain_file = "data/ace05_en.raw.choice.ch.process"
CH_train_file = "data/ace05_ch.choice.train"
CH_test_file = "data/ace05_ch.choice.test"

'''
EN_train_file = "data/ace05_en.sst.example"
CH_train_file = "data/ace05_ch.train.sst.example"
CH_test_file = "data/ace05_ch.test.sst.example"
'''
#pkl path
data_pkl = 'data/pdata.pkl'
emb_pkl = 'data/pemb_weight.pkl'

chconv_model = 'cache/chconv.pkl'
h2tag_model = 'cache/h2tag.pkl'
chword_emb_model = 'cache/chword.pkl'
chdist_emb_model = 'cache/chdist.pkl'
chpos_emb_model = 'cache/chpos.pkl'
chner_emb_model = 'cache/chner.pkl'

Merge_gan_model = "result/gan22.model"
Merge_scnn_model = "data/scnn_model.pkl"
