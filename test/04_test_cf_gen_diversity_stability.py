from os.path import join

from evomol.evaluation_dft import smi_to_filename

from xaimol import generate_black_box_function, generate_counterfactuals, get_distance_function
from xaimol.classifier import BlackBoxClassifier

import os
import pandas as pd
import numpy as np
import json
import joblib
from chemdesc import ShinglesVectDesc

from xaimol.visualization import plot_counterfactuals

model_path = os.environ["DATA"] + "/08_XAI/01.01_stability_learning/model.pkl"
test_set_path = os.environ["DATA"] + "/00_datasets/DFT/stability/test_set.csv"
all_OD9_shingles_dict_path_own = os.environ["DATA"] + "/00_datasets/DFT/stability/all_OD9_shingles_dict_own.json"


def load_dataset(csv_path, first_col=None):
    df = pd.read_csv(csv_path, header=None)
    if first_col is not None:
        df.drop(np.arange(0, first_col), axis=1, inplace=True)

    return df.to_numpy()


def extract_smiles_to_study(test_set_data, black_box_classifier, expected_class_test_data, expected_class_predicted):
    smiles_to_study = []
    for i in range(1, 21):
        curr_smi = test_set_data[i][1]
        smiles_to_study.append(curr_smi)

    return smiles_to_study


model_rf = joblib.load(model_path)
test_set_data = load_dataset(test_set_path)

with open(all_OD9_shingles_dict_path_own, "r") as f:
    shingles_dict = json.load(f)

print(test_set_data)

desc_builder_g = ShinglesVectDesc(vect_size=1000, external_desc_id_dict=shingles_dict)


black_box_classifier = generate_black_box_function(model_rf, descriptor=desc_builder_g)

smiles_to_study = extract_smiles_to_study(test_set_data, black_box_classifier, '1', 'positive')

results_root = "test/04_results"

paths = []
distance_functions = []
for smi in smiles_to_study[:5]:
    path = join(results_root, smi_to_filename(smi))
    paths.append(path)
    dist_fun = get_distance_function("tanimoto_ecfp4", smi)
    distance_functions.append(dist_fun)
    # generate_counterfactuals(smi, black_box_classifier, path, entropy_key="entropy_gen_scaffolds", entropy_weight=1)

plot_counterfactuals(smiles_to_study[:5], paths, black_box_classifier, distance_functions=distance_functions,
                     fig_save_path="test/04_results/fig.png", n_counterfactuals=5, plot_modification_map=True)
