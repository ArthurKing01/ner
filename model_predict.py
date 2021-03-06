import kashgari
import argparse
import os
from dirs import keras_models_dir

parser = argparse.ArgumentParser(description="your script description") 
parser.add_argument('--model_name', '-n', required=True, type=str)
args = parser.parse_args()

model_name = args.model_name
model_path = os.path.join(keras_models_dir, model_name)

loaded_model = kashgari.utils.load_model(model_path)
loaded_model.tf_model.summary()
while True:
    text = input('sentence: ')
    r = loaded_model.predict([[char for char in text]])
    print(r)
    per, loc, org = '', '', ''

    for i, t in enumerate(r[0]):
        if t in ('B-PER', 'I-PER', '\tI-PER'):
            per += ',' + text[i] if (t == 'B-PER' and not per == '') else text[i]
        if t in ('B-ORG', 'I-ORG'):
            org += ',' + text[i] if (t == 'B-ORG' and not org == '') else text[i]
        if t in ('B-LOC', 'I-LOC'):
            loc += ',' + text[i] if (t == 'B-LOC' and not loc == '') else text[i]

    print(['person: ' + per, 'location: ' + loc, 'organzation: ' + org])
