# -*- coding: utf-8 -*-
# time: 2019-08-09 16:47
# place: Zhichunlu Beijing

import argparse



import kashgari
from kashgari.corpus import DataReader
from kashgari.embeddings import BertEmbedding
from kashgari.tasks.labeling import BiLSTM_CRF_Model, BiLSTM_Model, BiGRU_Model

parser = argparse.ArgumentParser(description="your script description") 
parser.add_argument('--model_path', '-m', required=False, type=str)
parser.add_argument('--bert', '-b', required=False, type=bool)
parser.add_argument('--epochs', '-e', required=False, type=int)
parser.add_argument('--batch_size', '-bs', required=False, type=int)
args = parser.parse_args()

model_path = args.model_path if args.model_path else 'per_ner.h5'
epochs = args.epochs if args.epochs else 1
batch_size = args.batch_size if args.batch_size else 16

# train_x, train_y = DataReader().read_conll_format_file('./data/cluener_public/train.data')
# valid_x, valid_y = DataReader().read_conll_format_file('./data/cluener_public/dev.data')
train_x, train_y = DataReader().read_conll_format_file('./data/train_data.data')
valid_x, valid_y = DataReader().read_conll_format_file('./data/test_data.data')
# test_x, test_y = DataReader().read_conll_format_file('./data/time.test')

if args.bert:
    bert_embedding = BertEmbedding('chinese_wwm_ext_L-12_H-768_A-12',
                               task='labeling',
                               sequence_length=128)

    # model = BiGRU_Model(bert_embedding)    
    model = BiLSTM_Model(bert_embedding)    
    # model = BiLSTM_CRF_Model(bert_embedding)    
else:
    # model = BiGRU_Model()
    model = BiLSTM_Model()
    # model = BiLSTM_CRF_Model()


model.fit(train_x, train_y, valid_x, valid_y, batch_size=batch_size, epochs=epochs)

model.save(model_path)

# model.evaluate(test_x, test_y)