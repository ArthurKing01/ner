# Load saved model
import kashgari
import argparse

parser = argparse.ArgumentParser(description="your script description") 
parser.add_argument('--model_path', '-m', required=False, type=str)
args = parser.parse_args()

model_path = args.model_path if args.model_path else 'per_ner.h5'

loaded_model = kashgari.utils.load_model(model_path)

while True:
    text = input('sentence: ')
    r = loaded_model.predict([[char for char in text]])
    print(r)
    per, loc, org = '', '', ''

    for i, t in enumerate(r[0]):
        if t in ('B-PER', 'I-PER'):
            per += ',' + text[i] if (t == 'B-PER' and not per == '') else text[i]
        if t in ('B-ORG', 'I-ORG'):
            org += ',' + text[i] if (t == 'B-ORG' and not org == '') else text[i]
        if t in ('B-LOC', 'I-LOC'):
            loc += ',' + text[i] if (t == 'B-LOC' and not loc == '') else text[i]

    print(['person: ' + per, 'location: ' + loc, 'organzation: ' + org])
