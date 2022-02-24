from abc import ABC, abstractmethod

from chemdesc import Descriptor


def flip_class_keyword(class_keyword):
    """
    Flipping the class keyword that is given as input.
    :param class_keyword: class keyword ("positive" or "negative") that must be flipped
    :return: str keyword ("positive" or "negative")
    """

    return "negative" if class_keyword == "positive" else "positive"


def class_keyword_to_boolean(class_keyword):
    """
    Converting the given class keyword into a boolean (True/False) value
    :param class_keyword:
    :return:
    """
    return True if class_keyword == "positive" else False


def class_keyword_to_integer_boolean(class_keyword):
    """
    Converting the given class keyword into an integer boolean (1/0) value
    :param class_keyword:
    :return:
    """
    return 1 if class_keyword == "positive" else 0


def class_keyword_to_signed_integer(class_keyword):
    """
    Converting the given class keyword into a signed integer (1/-1) value
    :param class_keyword:
    :return:
    """
    return 1 if class_keyword == "positive" else -1


class BlackBoxClassifier(ABC):
    """
    Class that represents a black box function.
    """

    def __init__(self, decision_frontier=0.5, flip_frontier=False, descriptor=None):
        """
        :param decision_frontier: decision frontier. Any sample assessed by the black box function at a value
        that is greater or equal (except if flip_frontier is set to True) than this threshold is considered positive.
        :param: flip_frontier : whether to flip the decision frontier. If set to True, then the samples that are
        strictly lesser to the decision frontier will be considered positive.
        :param descriptor : instance of chemdesc.descriptors.Descriptor that is used to convert SMILES into a descriptor
        that is used as input of the black box classifier. If None, SMILES are given directly
        """
        self.decision_frontier = decision_frontier
        self.flip_frontier = flip_frontier

        self.descriptor = descriptor

        if isinstance(descriptor, Descriptor):
            descriptor.disable_tqdm = True

    @abstractmethod
    def _compute_proba_value(self, black_box_input):
        """
        Computing the [0, 1] probability value assuming that 0 is negative class and 1 is positive class
        :param black_box_input: black box input of the molecule to be assessed
        :return:
        """

    def _get_black_box_input(self, smiles):
        """
        Converting a SMILES into an input that is accepted by the black box classifier (using the self.descriptor
        instance if not None)
        :param smiles: molecule to be converted
        :return:
        """

        return smiles if self.descriptor is None else self.descriptor.fit_transform([smiles])[0][0]

    def assess_proba_value(self, smiles, target_class="positive"):
        """
        Computing the [0, 1] probability that the given instance is assessed as of the target class according to the
        black box classifier.
        :param smiles:
        :param target_class:
        :return:
        """
        computed_value = self._compute_proba_value(self._get_black_box_input(smiles))
        positive_value = computed_value if not self.flip_frontier else 1 - computed_value

        if target_class == "positive":
            return positive_value
        else:
            return 1 - positive_value

    def get_decision_frontier(self, target_class="positive"):
        """
        Returning the decision frontier such as a sample that is predicted with a value that is above this threshold is
        considered of target class
        :param target_class:
        :return:
        """
        decision_frontier_positive = self.decision_frontier if not self.flip_frontier else 1 - self.decision_frontier

        if target_class == "positive":
            return decision_frontier_positive
        else:
            return 1 - decision_frontier_positive

    def assess_class(self, smiles):
        """
        Returning the class keyword of the input SMILES according to the black box classifier
        :param smiles:
        :return:
        """
        if self.assess_proba_value(smiles, "positive") >= self.get_decision_frontier("positive"):
            return "positive"
        else:
            return "negative"


class CustomBlackBoxClassifier(BlackBoxClassifier):
    """
    Wrapping a custom python function evaluating any molecule by returning a value between 0 (negative class) and 1
    (positive class).
    """

    def __init__(self, function, decision_frontier=0.5, flip_frontier=False, descriptor=None):
        """
        :param function python function that returns a value between 0 (negative class) and 1 (positive class)
        :param decision_frontier: decision frontier. Any sample assessed by the black box function at a value
        that is greater or equal (except if flip_frontier is set to True) than this threshold is considered positive.
        :param: flip_frontier : whether to flip the decision frontier. If set to True, then the samples that are
        strictly lesser to the decision frontier will be considered positive.
        :param descriptor : instance of chemdesc.descriptors.Descriptor that is used to convert SMILES into a descriptor
        that is used as input of the black box classifier. If None, SMILES are given directly
        """

        super().__init__(decision_frontier, flip_frontier, descriptor)
        self.classifier = function

    def _compute_proba_value(self, black_box_input):
        return self.classifier(black_box_input)


class SKLearnBlackBoxClassifier(BlackBoxClassifier):
    """
    Wrapping any scikit learn classifier as a XAIMol.classifier.BlackBoxClassifier
    """

    def __init__(self, classifier, decision_frontier=0.5, flip_frontier=False, descriptor=None):
        """
        :param scikit learn classifier
        :param decision_frontier: decision frontier. Any sample assessed by the black box function at a value
        that is greater or equal (except if flip_frontier is set to True) than this threshold is considered positive.
        :param: flip_frontier : whether to flip the decision frontier. If set to True, then the samples that are
        strictly lesser to the decision frontier will be considered positive.
        :param descriptor : instance of chemdesc.descriptors.Descriptor that is used to convert SMILES into a descriptor
        that is used as input of the black box classifier. If None, SMILES are given directly
        """
        super().__init__(decision_frontier, flip_frontier, descriptor)
        self.classifier = classifier

    def _compute_proba_value(self, black_box_input):
        # Returning the probability of the positive (=second) class according to the classifier
        return self.classifier.predict_proba(black_box_input.reshape(1, -1))[0][1]


class EvoMolEvaluationStrategyClassifier(BlackBoxClassifier):
    """
    Wrapping an evomol.evaluation.EvaluationStrategyComposant instance
    """

    def __init__(self, classifier, decision_frontier=0.5, flip_frontier=False, descriptor=None):
        """
        :param classifier: evomol.evaluation.EvaluationStrategyComposant instance
        :param decision_frontier: decision frontier. Any sample assessed by the black box function at a value
        that is greater or equal (except if flip_frontier is set to True) than this threshold is considered positive.
        :param: flip_frontier : whether to flip the decision frontier. If set to True, then the samples that are
        strictly lesser to the decision frontier will be considered positive.
        :param descriptor : instance of chemdesc.descriptors.Descriptor that is used to convert SMILES into a descriptor
        that is used as input of the black box classifier. If None, SMILES are given directly
        """
        super().__init__(decision_frontier, flip_frontier, descriptor)
        self.classifier = classifier

    def _compute_proba_value(self, black_box_input):
        return self.classifier.eval_smi(black_box_input)
