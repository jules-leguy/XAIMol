from evomol import EvaluationStrategy
from guacamol.common_scoring_functions import TanimotoScoringFunction
from rdkit.Chem import MolFromSmiles

from .classifier import flip_class_keyword


class ECFP4TanimotoSimilarity(EvaluationStrategy):
    """
    Evaluating the Tanimoto similarity on ECFP4 fingerprints
    """

    def __init__(self, target_smiles):
        super().__init__()
        self.target_smiles = target_smiles
        self.guacamol_scorer = TanimotoScoringFunction(target_smiles, "ECFP4")

    def keys(self):
        return ["ecfp4_tanimoto"]

    def evaluate_individual(self, individual, to_replace_idx=None):
        value = self.guacamol_scorer.score_mol(MolFromSmiles(individual.to_aromatic_smiles()))
        return value, [value]


class FlipClassObjective(EvaluationStrategy):
    """
    Objective function that orients the search towards solutions that are assessed by the black box function as of the
    opposite class than the original sample for which a counterfactual is searched.
    """

    def __init__(self, black_box_classifier, original_class="positive"):

        super().__init__()
        self.black_box_classifier = black_box_classifier
        self.original_class = original_class

    def keys(self):
        return ["model_proba_obj"]

    def evaluate_individual(self, individual, to_replace_idx=None):

        # Computing the probability that the class is flipped
        proba_flip = self.black_box_classifier.assess_proba_value(individual.to_aromatic_smiles(),
                                                                  target_class=flip_class_keyword(self.original_class))

        # Computing the decision frontier above which the class is flipped
        decision_frontier = self.black_box_classifier.get_decision_frontier(
            target_class=flip_class_keyword(self.original_class))

        # Computing the objective value
        val = min(1, 1/decision_frontier * proba_flip)

        return val, [val]
