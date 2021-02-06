import pickle
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import normalize


def calculate_angle(row):
    return np.arcsin(row['z_difference'] / row['distance'])


def calculate_distance(row):
    distance = np.linalg.norm(np.array(row['start_point']) - np.array(row['end_point']))
    return distance


# In case there are not all team/map/grenade_type in test data
def add_missing_dummy_columns(df):
    required_columns = ['team_CT', 'team_T', 'TYPE_flashbang', 'TYPE_molotov',
                        'TYPE_smoke', 'map_name_de_inferno', 'map_name_de_mirage']
    missing_cols = set(required_columns) - set(df.columns)
    for c in missing_cols:
        df[c] = 0


def prepare_dataset(dataset):
    dataset = dataset.copy()
    dataset['throw_time'] = dataset['detonation_tick'] - dataset['throw_tick']
    for label in ['x', 'y', 'z']:
        dataset[f'{label}_difference'] = dataset[f'detonation_raw_{label}'] - dataset[f'throw_from_raw_{label}']
    dataset['start_point'] = dataset[['throw_from_raw_x', 'throw_from_raw_y', 'throw_from_raw_z']].to_numpy().tolist()
    dataset['end_point'] = dataset[['detonation_raw_x', 'detonation_raw_y', 'detonation_raw_z']].to_numpy().tolist()
    dataset['distance'] = dataset.apply(calculate_distance, axis=1)
    dataset['angle'] = dataset.apply(calculate_angle, axis=1)
    X = dataset[['team', 'detonation_raw_x', 'detonation_raw_y', 'detonation_raw_z', 'throw_from_raw_x',
                 'throw_from_raw_y', 'throw_from_raw_z', 'throw_time', 'distance', 'y_difference', 'x_difference',
                 'z_difference', 'TYPE', 'angle', 'map_name']]
    X = pd.get_dummies(X, columns=['team', 'TYPE', 'map_name'])
    add_missing_dummy_columns(X)
    X = normalize(X, norm='l2')
    return X


def join_files(files):
    dfs = []
    for file in files:
        df = pd.read_csv(file, index_col=None, header=0)
        dfs.append(df)
    return pd.concat(dfs, axis=0, ignore_index=True)


def create_model(files):
    dataset = join_files(files)
    X = prepare_dataset(dataset)
    y = dataset.LABEL.values
    oversample = SMOTE()
    X, y = oversample.fit_resample(X, y)
    model = RandomForestClassifier(n_estimators=1000, max_features=19)
    model.fit(X, y)
    pickle.dump(model, open('grenade_model', 'wb'))


if __name__ == '__main__':
    create_model(['train-grenades-de_mirage.csv', 'train-grenades-de_inferno.csv'])
