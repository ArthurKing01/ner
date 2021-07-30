# -*- coding: utf-8 -*-
# time: 2019-08-09 16:47
# place: Zhichunlu Beijing

import kashgari
from kashgari.corpus import DataReader
from kashgari.embeddings import BertEmbedding
from kashgari.tasks.labeling import BiLSTM_CRF_Model

train_x, train_y = DataReader().read_conll_format_file('./data/train_data.data')
valid_x, valid_y = DataReader().read_conll_format_file('./data/test_data.data')
# test_x, test_y = DataReader().read_conll_format_file('./data/time.test')

# bert_embedding = BertEmbedding('chinese_wwm_ext_L-12_H-768_A-12',
#                                task='labeling',
#                                sequence_length=128)

model = BiLSTM_CRF_Model()
model.fit(train_x, train_y, valid_x, valid_y, batch_size=16, epochs=5)

model.save('bilstm_per_ner.h5')

# model.evaluate(test_x, test_y)