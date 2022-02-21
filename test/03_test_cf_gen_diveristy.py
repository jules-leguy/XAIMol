from os.path import join

from evomol.evaluation_dft import smi_to_filename
from rdkit.Chem import rdqueries, MolFromSmiles


# Negative : less than 50% heteroatoms
# Positive : more than 50% heteroatoms


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

black_box_classifier = generate_black_box_function(carbon_proportion, flip_frontier=True, decision_frontier=0.5)


results_root = "test/03_results"

paths = []

for smi in test_set:
    path = join(results_root, smi_to_filename(smi))
    paths.append(path)
    # generate_counterfactuals(smi, black_box_classifier, path, entropy_key="entropy_ecfp4", entropy_weight=1)

plot_counterfactuals(test_set, paths, black_box_classifier, "test/03_results/fig.png", n_counterfactuals=5)
