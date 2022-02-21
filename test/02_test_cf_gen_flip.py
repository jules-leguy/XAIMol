from os.path import join

from evomol.evaluation_dft import smi_to_filename
from rdkit.Chem import rdqueries, MolFromSmiles


# Negative : less than 30% heteroatoms
# Positive : more than 30% heteroatoms


from xaimol import generate_black_box_function, generate_counterfactuals
from xaimol.classifier import BlackBoxClassifier
from xaimol.visualization import plot_counterfactuals


def carbon_proportion(smiles):
    mol = MolFromSmiles(smiles)
    q = rdqueries.AtomNumEqualsQueryAtom(6)
    n_carbon = len(mol.GetAtomsMatchingQuery(q))
    n_atoms = mol.GetNumAtoms()
    return n_carbon / n_atoms


test_set = ["CNF", "CC(=O)OC1=CC=CC=C1C(=O)O", "CCCC", "CCCN", "c1ccc(cc1)Cl"]

black_box_classifier = generate_black_box_function(carbon_proportion, flip_frontier=True, decision_frontier=0.3)

print(black_box_classifier.assess_class("CNF"))



results_root = "test/02_results"

paths = []

for smi in test_set:
    path = join(results_root, smi_to_filename(smi))
    paths.append(path)
    # generate_counterfactuals(smi, black_box_classifier, path)

plot_counterfactuals(test_set, paths, black_box_classifier, "test/02_results/fig.png")
