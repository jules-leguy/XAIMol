from os.path import join

import numpy as np

from xaimol.classifier import flip_class_keyword
import pandas as pd


def extract_counterfactuals_results(target_smiles, experiment_path, black_box_classifier, distance_function):
    """
    Returning the list of counterfactuals obtained by the given experiment. The results are sorted by decreasing
    similarity. Also returning the similarity vector of same size as the number of returned solutions. If the distance
    function is the same as the one used during optimization then its values are not re-computed.

    :param target_smiles: SMILES of the molecule for which counterfactuals explanations were searched
    :param experiment_path: path were the experiment results are stored
    :param black_box_classifier: xaimol.classifier.BlackBoxClassifier instance that corresponds to the black box
    function that is being explained.
    :param distance_function: distance function (see xaimol.get_distance_function function).
    :return: list of smiles, list of black box classifier predicted values, list of similarity values
    """

    # Computing the expected class of counterfactual explanations
    counterfactuals_class = flip_class_keyword(black_box_classifier.assess_class(target_smiles))

    # Reading EvoMol population
    df = pd.read_csv(join(experiment_path, "pop.csv"))

    # Checking if distance function differs from the one used during optimization
    distance_function_differs = distance_function.keys()[0] not in df

    # Removing nan values in columns where relevant (population not full)
    if distance_function_differs:
        df.dropna(subset=["smiles"], inplace=True)
    else:
        df.dropna(subset=["smiles", distance_function.keys()[0]], inplace=True)

    # Extracting SMILES list of resulting individuals in the population
    smiles_list_pop = df["smiles"].tolist()

    # Initialisation of data structures
    cf_smiles_list, cf_predicted_values, cf_sim_values = [], [], []

    # Iterating over all solutions
    for i, curr_sol_smi in enumerate(smiles_list_pop):

        # Computing predicted class for current solution
        predicted_class = black_box_classifier.assess_class(curr_sol_smi)

        # If the current solution is a counterfactual explanation
        if predicted_class == counterfactuals_class:

            # Computing the exact predicted value
            predicted_value = black_box_classifier.assess_proba_value(curr_sol_smi)

            # Computing or extracting from data the similarity value
            if distance_function_differs:
                sim_value = distance_function.eval_smi(curr_sol_smi)
            else:
                sim_value = df[distance_function.keys()[0]][i]

            # Saving values to corresponding data structures
            cf_smiles_list.append(curr_sol_smi)
            cf_predicted_values.append(predicted_value)
            cf_sim_values.append(sim_value)

    # Sorting all lists by decreasing similarity values
    sort_mask = np.argsort(cf_sim_values)[::-1]
    cf_smiles_list = np.array(cf_smiles_list)[sort_mask]
    cf_predicted_values = np.array(cf_predicted_values)[sort_mask]
    cf_sim_values = np.array(cf_sim_values)[sort_mask]

    return cf_smiles_list, cf_predicted_values, cf_sim_values

