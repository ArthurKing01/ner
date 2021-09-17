import argparse
import tensorflow as tf
from kashgari.corpus import DataReader
from kashgari.embeddings import BertEmbedding
from kashgari.tasks.labeling import BiLSTM_CRF_Model, BiLSTM_Model
import os
from dirs import keras_models_dir, data_dir

parser = argparse.ArgumentParser(description="your script description") 
parser.add_argument('--model_name', '-n', required=True, type=str)
parser.add_argument('--bert', '-b', required=False, type=bool)
parser.add_argument('--crf', '-c', required=False, type=bool)
parser.add_argument('--epochs', '-e', required=False, type=int)
parser.add_argument('--batch_size', '-bs', required=False, type=int)
args = parser.parse_args()

print('args:', args)

epochs = args.epochs if args.epochs else 1
batch_size = args.batch_size if args.batch_size else 64


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
  except RuntimeError as e:
    # Visible devices must be set at program startup
    print(e)


train_x, train_y = DataReader().read_conll_format_file(os.path.join(data_dir, 'time.train'))
valid_x, valid_y = DataReader().read_conll_format_file(os.path.join(data_dir, 'time.test'))

if args.bert:
    bert_embedding = BertEmbedding('chinese_wwm_ext_L-12_H-768_A-12',
                               task='labeling',
                               sequence_length=128)
 
    if args.crf:
      model = BiLSTM_CRF_Model(bert_embedding)
    else:
      model = BiLSTM_Model(bert_embedding)        
else:
    if args.crf:
      model = BiLSTM_CRF_Model()
    else:
      model = BiLSTM_Model()

model.fit(train_x, train_y, valid_x, valid_y, batch_size=batch_size, epochs=epochs)


model_name = args.model_name
model_path = os.path.join(keras_models_dir, model_name)
model.save(model_path)
