import pickle
import pandas as pd
import argparse
from model_generator import prepare_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    args = parser.parse_args()
    model = pickle.load(open('grenade_model', 'rb'))
    dataset = pd.read_csv(args.filename)
    X = prepare_dataset(dataset)
    dataset['RESULT'] = model.predict(X)
    dataset['RESULT'] = dataset['RESULT'].apply(lambda x: str(x).upper())
    dataset.to_csv(args.filename, index=False)
