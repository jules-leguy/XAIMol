import os

from IPython.core.display_functions import display

model_path = os.environ["DATA"] + "/08_XAI/01.01_stability_learning/model.pkl"
test_set_path = os.environ["DATA"] + "/00_datasets/DFT/stability/test_set.csv"
all_OD9_shingles_dict_path_own = os.environ["DATA"] + "/00_datasets/DFT/stability/all_OD9_shingles_dict_own.json"

output_optim_path = os.environ["DATA"] + "/08_XAI/01.01_stability_learning/experiments_egc_workshop"

import pandas as pd


def load_dataset(csv_path, first_col=None):
    df = pd.read_csv(csv_path, header=None)
    if first_col is not None:
        df.drop(np.arange(0, first_col), axis=1, inplace=True)

    return df.to_numpy()


import json
import joblib

model_rf = joblib.load(model_path)
test_set_data = load_dataset(test_set_path)

with open(all_OD9_shingles_dict_path_own, "r") as f:
    shingles_dict = json.load(f)

print(test_set_data)

from chemdesc import ShinglesVectDesc
desc_builder_g = ShinglesVectDesc(vect_size=1000, external_desc_id_dict=shingles_dict)

from evomol.evaluation import EvaluationStrategy, EvaluationStrategyComposant
import numpy as np
from guacamol.common_scoring_functions import TanimotoScoringFunction


class ShinglesTanimotoSimilarity(EvaluationStrategy):

    def __init__(self, target_smiles, desc_builder):
        super().__init__()
        self.target_smiles = target_smiles
        self.desc_builder = desc_builder
        self.target_desc = desc_builder.fit_transform([target_smiles])[0][0].astype(bool)

    def keys(self):
        return ["shingles_tanimoto"]

    def evaluate_individual(self, individual, to_replace_idx=None):
        candidate_desc = self.desc_builder.fit_transform([individual.to_aromatic_smiles()])[0][0].astype(bool)
        intersect = np.logical_and(self.target_desc, candidate_desc)
        value = np.sum(intersect) / (np.sum(self.target_desc) + np.sum(candidate_desc) - np.sum(intersect))
        return value, [value]


class ECFP4TanimotoSimilarity(EvaluationStrategy):

    def __init__(self, target_smiles):
        super().__init__()
        self.target_smiles = target_smiles
        self.guacamol_scorer = TanimotoScoringFunction(target_smiles, "ECFP4")

    def keys(self):
        return ["ecfp4_tanimoto"]

    def evaluate_individual(self, individual, to_replace_idx=None):
        value = self.guacamol_scorer.score_mol(MolFromSmiles(individual.to_aromatic_smiles()))
        return value, [value]


def predict_smiles(smi, model, desc_builder):
    desc = desc_builder.fit_transform([smi])[0][0]
    return model.predict_proba(desc.reshape(1, -1))[0][0]


class ModelProbabilityObjective(EvaluationStrategy):

    def __init__(self, model, desc_builder):
        super().__init__()
        self.model = model
        self.desc_builder = desc_builder

    def keys(self):
        return ["model_proba_obj"]

    def evaluate_individual(self, individual, to_replace_idx=None):
        proba_neg = predict_smiles(individual.to_aromatic_smiles(), self.model, self.desc_builder)
        val = min(1, 2 * proba_neg)
        return val, [val]


from evomol.evaluation_dft import smi_to_filename
from os.path import join
from rdkit.Chem.Draw import MolsToGridImage, DrawingOptions
from rdkit.Chem import MolFromSmiles
from IPython.utils import io
from evomol import run_model
from PIL import Image, ImageDraw


def run_exp(smi, only_change_bond_substitute_atom):
    with io.capture_output() as captured:

        sim_obj = ECFP4TanimotoSimilarity(smi)
        proba_obj = ModelProbabilityObjective(model_rf, desc_builder_g)

        obj_fun = {
            "type": "mean",
            "functions": [
                sim_obj,
                proba_obj
            ]
        }

        if only_change_bond_substitute_atom:
            restriction_str = "restriction"
        else:
            restriction_str = "no_restriction"

        action_space_parameters = {
            "atoms": "C,N,O,F",
            "append_atom": not only_change_bond_substitute_atom,
            "remove_atom": not only_change_bond_substitute_atom,
            "cut_insert": not only_change_bond_substitute_atom,
            "move_group": not only_change_bond_substitute_atom,
            "change_bond_prevent_breaking_creating_bonds": only_change_bond_substitute_atom
        }

        exp_dict = {
            "obj_function": obj_fun,
            "io_parameters": {
                "model_path": join(output_optim_path, restriction_str + "-" + smi_to_filename(smi)),
                "smiles_list_init": [smi]
            },
            "action_space_parameters": action_space_parameters,
            "optimization_parameters": {
                "max_steps": 100,
                "mutation_max_depth": 1
            }
        }

        run_model(exp_dict)


def plot_results(smi, only_change_bond_substitute_atom):
    with io.capture_output() as captured:

        sim_obj = ECFP4TanimotoSimilarity(smi)
        proba_obj = ModelProbabilityObjective(model_rf, desc_builder_g)

        obj_fun = {
            "type": "mean",
            "functions": [
                sim_obj,
                proba_obj
            ]
        }

        if only_change_bond_substitute_atom:
            restriction_str = "restriction"
        else:
            restriction_str = "no_restriction"

        df = pd.read_csv(join(output_optim_path, restriction_str + "-" + smi_to_filename(smi), "pop.csv"))

        # Keeping only solutions that are predicted negative
        df = df[df["model_proba_obj"] == 1]

        # Sorting by descending similarity
        df = df.sort_values("total", ascending=False)

        selected_solutions = df["smiles"][:3].tolist()
        selected_solutions.insert(0, smi)

        predictions = [str("{:.2f}".format(predict_smiles(smi, model_rf, desc_builder_g))) for smi in
                       selected_solutions]
        str_class = ["+" for i in range(len(selected_solutions))]
        str_class[0] = "-"
        sim = [str("{:.2f}".format(sim_obj.eval_smi(smi))) for smi in selected_solutions]

        legends = [str_class[i] + ", " + predictions[i] + ", " + sim[i] for i in range(len(selected_solutions))]

    img = MolsToGridImage([MolFromSmiles(s) for s in selected_solutions], legends=legends, returnPNG=False,
                          molsPerRow=4, subImgSize=(250, 250))

    with open(join(output_optim_path, restriction_str + "-" + smi_to_filename(smi), "img.png"), "wb") as f:
        img.save(f, "png")

    return img


def plot_results_subset(smiles_list, idx_list, only_change_bond_substitute_atom):
    all_selected_solutions = []
    all_predictions = []
    all_str_class = []
    all_sim = []

    for idx in idx_list:

        smi = smiles_list[idx]

        with io.capture_output() as captured:

            sim_obj = ECFP4TanimotoSimilarity(smi)
            proba_obj = ModelProbabilityObjective(model_rf, desc_builder_g)

            obj_fun = {
                "type": "mean",
                "functions": [
                    sim_obj,
                    proba_obj
                ]
            }

            if only_change_bond_substitute_atom:
                restriction_str = "restriction"
            else:
                restriction_str = "no_restriction"

            df = pd.read_csv(join(output_optim_path, restriction_str + "-" + smi_to_filename(smi), "pop.csv"))

            # Keeping only solutions that are predicted negative
            df = df[df["model_proba_obj"] == 1]

            # Sorting by descending similarity
            df = df.sort_values("total", ascending=False)

            selected_solutions = df["smiles"][:3].tolist()
            selected_solutions.insert(0, smi)

            predictions = [str("{:.2f}".format(predict_smiles(smi, model_rf, desc_builder_g))) for smi in
                           selected_solutions]
            str_class = ["+" for i in range(len(selected_solutions))]
            str_class[0] = "-"
            sim = [str("{:.2f}".format(sim_obj.eval_smi(smi))) for smi in selected_solutions]

            all_selected_solutions.extend(selected_solutions)
            all_predictions.extend(predictions)
            all_str_class.extend(str_class)
            all_sim.extend(sim)

    legends = [all_str_class[i] + ", " + all_predictions[i] + ", " + all_sim[i] for i in
               range(len(all_selected_solutions))]

    img = MolsToGridImage([MolFromSmiles(s) for s in all_selected_solutions], legends=legends, returnPNG=False,
                          molsPerRow=4, subImgSize=(200, 200))

    with open(join(output_optim_path, str(idx_list) + "_" + restriction_str + ".png"), "wb") as f:
        img.save(f, "png")

    return img


def extract_smiles_to_study(test_set_data):
    smiles_to_study = []
    for i in range(1, 21):
        curr_smi = test_set_data[i][1]
        if test_set_data[i][2] == '1' and predict_smiles(curr_smi, model_rf, desc_builder_g) < 0.5:
            smiles_to_study.append(curr_smi)

    return smiles_to_study


smiles_to_study = extract_smiles_to_study(test_set_data)
print(len(smiles_to_study))
print(smiles_to_study)

from xaimol.visualization import plot_counterfactuals
from xaimol import generate_black_box_function

with io.capture_output() as captured:
    smiles_to_study = extract_smiles_to_study(test_set_data)

smiles_to_study = smiles_to_study[:2]

black_box_classifier = generate_black_box_function(model_rf, descriptor=desc_builder_g, flip_frontier=True)


def viz_xaimol(smiles_to_study, restriction_str, black_box_classifier):
    paths = [join(output_optim_path, restriction_str + "-" + smi_to_filename(smi)) for smi in smiles_to_study]

    display(plot_counterfactuals(smiles_to_study, paths, black_box_classifier, n_counterfactuals=2,
                                 mol_size=(200, 200), fig_save_path="/home/jleguy/test_cf.png"))

viz_xaimol(smiles_to_study, "no_restriction", black_box_classifier)