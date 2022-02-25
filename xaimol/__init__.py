from evomol import run_model
from evomol.evaluation import EvaluationStrategyComposant, NPerturbationsEvaluationStrategy, \
    OppositeWrapperEvaluationStrategy
from sklearn.base import ClassifierMixin

from .classifier import SKLearnBlackBoxClassifier, CustomBlackBoxClassifier, EvoMolEvaluationStrategyClassifier
from .objective import ECFP4TanimotoSimilarity, FlipClassObjective


def get_distance_function(similarity_function, target_smiles):
    """
    Returning the EvoMol.evaluation.EvaluationStrategyComposant instance that evaluates the distance between any
    molecule and the smiles given as parameter (target_smiles). In the special case of using a distance as a penalty
    of the number of mutations, the distance is calculated with respect to the first individual of the population.
    :param similarity_function: distance function to be used. Can be either "tanimoto_ecfp4" in order to use the
    Tanimoto distance on ECFP4 fingerprints, or can be "number_mutations_penalty" to compute a penalty that is based
    on the number of previously performed mutations. It can finally also be an
    EvoMol.evaluation.EvaluationStrategyComposant class that takes as constructor argument the target SMILES.
    :param target_smiles: SMILES to evaluate the distance of instances with
    :return:
    """
    if similarity_function == "tanimoto_ecfp4":
        return ECFP4TanimotoSimilarity(target_smiles)
    if similarity_function == "number_mutations_penalty":
        return OppositeWrapperEvaluationStrategy([NPerturbationsEvaluationStrategy()])
    elif issubclass(similarity_function, EvaluationStrategyComposant):
        return similarity_function(target_smiles)
    else:
        return None


def get_class_objective_function(black_box_function, target_smiles):
    """
    Returning the XAIMol.objective.FlipClassObjective instance that guides the search towards solutions of opposite
    class compared to target_smiles with respect to the given black box function
    :param black_box_function: XAIMol.classifier.BlackBoxClassifier instance
    :param target_smiles: SMILES for which solutions of opposite class must be found
    :return: XAIMol.objective.FlipClassObjective instance
    """
    return FlipClassObjective(black_box_function, original_class=black_box_function.assess_class(target_smiles))


def _get_action_space_parameters_evomol(action_space_parameters, freeze_connectivity):
    """
    Returning the "action_space_parameters" input EvoMol attribute based on specified inputs.
    The action_space_parameters are used if they are not None. However, they can be overwritten if freeze_connectivity
    is set to True. The latter corresponds to whether the connectivity of the molecular graph is frozen (cannot be
    modified by mutations).
    :param action_space_parameters: dictionary input of EvoMol or None
    :param freeze_connectivity: whether the connectivity of the molecular graph must be frozen
    :return: action_space_parameters dictionary
    """

    action_space_parameters = action_space_parameters if action_space_parameters is not None else {}

    if freeze_connectivity:
        action_space_parameters["append_atom"] = not freeze_connectivity
        action_space_parameters["remove_atom"] = not freeze_connectivity
        action_space_parameters["cut_insert"] = not freeze_connectivity
        action_space_parameters["move_group"] = not freeze_connectivity
        action_space_parameters["change_bond_prevent_breaking_creating_bonds"] = freeze_connectivity

    return action_space_parameters


def _get_optimization_parameters_evomol(optimization_parameters):
    """
    Returning the "optimization_parameters" input EvoMol attribute based on specified input.
    If None, using default EvoMol paramters except for the max_steps attribute that is set to 100 instead of 1000. If
    not None, using the given input.
    :param optimization_parameters: None or dict value of "optimization_parameters" EvoMol input.
    :return: "optimization_parameters" dict
    """

    return {"max_steps": 100} if optimization_parameters is None else optimization_parameters


def generate_black_box_function(function, decision_frontier=0.5, flip_frontier=False, descriptor=None):
    """
    Returning a XAIMol.classifier.BlackBoxClassifier instance based on the given function.
    The given function can be either a [0, 1] function, a Scikit-Learn classifier instance or an EvoMol evaluation
    strategy instance.
    The returned object can be readily used as input of XAIMol.generate_counterfactuals
    :param function: classification function. Can be either a python function evaluating a single molecule, a
    Scikit-Learn classifier instance or an evomol.evaluation.EvaluationStrategyComposant instance.
    :param decision_frontier: decision frontier. Any sample assessed by the black box function at a value
    that is greater or equal (except if flip_frontier is set to True) than this threshold is considered positive.
    :param flip_frontier: whether to flip the decision frontier. If set to True, then the samples that are
    strictly lesser to the decision frontier will be considered positive.
    :param descriptor: keyword describing an instance of chemdesc.descriptors.Descriptor or
    chemdesc.descriptors.Descriptor instance. Available keywords are : "MBTR", "SOAP", "ShinglesVect" or "CoulombMatrix"
    :return: XAIMol.BlackBoxClassifier instance that can be used in XAIMol.generate_counterfactuals
    """

    if isinstance(function, ClassifierMixin):
        return SKLearnBlackBoxClassifier(function, decision_frontier,  flip_frontier, descriptor)
    elif isinstance(function, EvaluationStrategyComposant):
        return EvoMolEvaluationStrategyClassifier(function, decision_frontier, flip_frontier, descriptor)
    else:
        return CustomBlackBoxClassifier(function, decision_frontier, flip_frontier, descriptor)


def generate_counterfactuals(objective_function, smiles, results_path, freeze_connectivity=False,
                             action_space_parameters=None, optimization_parameters=None):
    """
    Generating counterfactuals for the given SMILES based on the given objective function
    :param objective_function: objective function of the counterfactuals generation procedure, using the EvoMol
    dictionary declaration syntax (https://github.com/jules-leguy/EvoMol). xaimol.get_distance_function and
    xaimol.get_class_objective_function functions can be used to respectively define the distance function and the
    function that guides the optimization towards solutions of the desired class.
    :param smiles: SMILES of the instance to be explained
    :param results_path: path to the folder that will contain the optimization results
    :param freeze_connectivity: whether the connectivity of the molecular graph of the smiles to be explained cannot be
    modified by the optimization (false)
    :param action_space_parameters: "action_space_parameters" attribute of the EvoMol optimization procedure. If None,
    default EvoMol parameters are used. Warning : some parameters can be overwritten by the freeze_connectivity
    parameter
    :param optimization_parameters: "optimization_parameters" attribute of the EvoMol optimization procedure. If None,
    default EvoMol parameters are used except for the 'max_steps' attribute that is set to 100.
    is not None.
    :return:
    """

    # Computing action_space_parameters dictionary
    action_space_parameters = _get_action_space_parameters_evomol(action_space_parameters, freeze_connectivity)

    # Computing optimization_parameters dictionary
    optimization_parameters = _get_optimization_parameters_evomol(optimization_parameters)

    # Computing experiment dictionary
    exp_dict = {
        "obj_function": objective_function,
        "io_parameters": {
            "model_path": results_path,
            "smiles_list_init": [smiles]
        },
        "action_space_parameters": action_space_parameters,
        "optimization_parameters": optimization_parameters
    }

    # Running EvoMol optimization
    run_model(exp_dict)
