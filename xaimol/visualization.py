import io
from os.path import join

import PIL
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt, pylab
from rdkit.Chem import MolFromSmiles, MolFromSmarts, Draw
from rdkit.Chem.Draw import MolsToGridImage, MolDrawing, DrawingOptions
from rdkit.Chem.Draw.SimilarityMaps import GetSimilarityMapFromWeights

from xaimol import _get_distance_function
from xaimol.postprocessing import extract_counterfactuals_results

from rdkit.Chem import rdFMCS as MCS


def _moldiff(template, query):
    """
    Implementation from exmol package : https://github.com/ur-whitelab/exmol

    MIT License

    Copyright (c) 2021 White Laboratory

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

    Compare the two rdkit molecules.
    :param template: template molecule
    :param query: query molecule
    :return: list of modified atoms in query, list of modified bonds in query
    """
    r = MCS.FindMCS([template, query])
    substructure = MolFromSmarts(r.smartsString)
    raw_match = query.GetSubstructMatches(substructure)
    template_match = template.GetSubstructMatches(substructure)

    if raw_match:
        # flatten it
        match = list(raw_match[0])
        template_match = list(template_match[0])

        # need to invert match to get diffs
        inv_match = [i for i in range(query.GetNumAtoms()) if i not in match]

        # get bonds
        bond_match = []
        for b in query.GetBonds():
            if b.GetBeginAtomIdx() in inv_match or b.GetEndAtomIdx() in inv_match:
                bond_match.append(b.GetIdx())

        # now get bonding changes from deletion
        def neigh_hash(a):
            return "".join(sorted([n.GetSymbol() for n in a.GetNeighbors()]))

        for ti, qi in zip(template_match, match):
            if neigh_hash(template.GetAtomWithIdx(ti)) != neigh_hash(
                    query.GetAtomWithIdx(qi)
            ):
                inv_match.append(qi)

        return inv_match, bond_match
    else:
        return [], []


def plot_mol_counterfactuals_modification_map(target_smiles, counterfactuals_smiles, legend, size=(250, 250),
                                              weights_threshold=0.):
    """
    Plotting a molecule and highlighting main differences with its counterfactual explanations.

    :param target_smiles: smiles of the molecule that is being explained
    :param counterfactuals_smiles: list of smiles of all the counterfactual explanations of target_smiles
    :param legend text of the legend that will be written below the molecule
    :param size: size of the output image in pixels. Default : (250, 250)
    :param weights_threshold: weights value below this threshold are ignored
    :return:
    """

    # Computing rdkit version of target mol
    target_mol = MolFromSmiles(target_smiles)

    # Initialization of the vector that counts the number of modifications of each atom
    modified_atoms_count = np.zeros((target_mol.GetNumAtoms(),))

    # Iterating over all counterfactual explanations
    for cf_smi in counterfactuals_smiles:

        # Computing the index of atoms that differ between the two molecules (in the referential of the target molecule)
        curr_cf_modified_atoms, _ = _moldiff(MolFromSmiles(cf_smi), target_mol)

        # Updating count vector
        for atom_idx in curr_cf_modified_atoms:
            modified_atoms_count[atom_idx] += 1

    # Normalization of the count into the weight vector
    weights = modified_atoms_count / modified_atoms_count.sum()
    # weights = weights * target_mol.GetNumAtoms()
    print(weights)

    weights[weights < weights_threshold] = 0

    # Computing output image
    fig = GetSimilarityMapFromWeights(target_mol, weights=weights, size=size, scale=-1, sigma=0.05,
                                      alpha=0, contourLines=0, step=0.01, coordScale=1)

    # Setting legend
    fig.text(1 / 2, y=0.0, s=legend, fontsize=30, horizontalalignment="center", transform=fig.gca().transAxes)

    return fig


def concatenate_images(img_list, horizontal_alignment=True):
    """
    Concatenation of all given images (horizontal or vertical).

    cf. https://note.nkmk.me/en/python-pillow-concat-images/

    MIT License

    Copyright (c) 2017

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

    :param img_list: list of PIL.Image.Image
    :param horizontal_alignment: whether the images are concatenated horizontally (default behaviour). If False,
    images are concatenated vertically.
    :return: PIL.Image.Image
    """

    # Returning empty image if list is empty
    if len(img_list) == 0:
        return Image.new('RGB', (0, 0))

    # If list contains several images, concatenation of the first image with the following images (recursive call)
    else:
        curr_img = img_list[0]
        following_img = concatenate_images(img_list[1:], horizontal_alignment=horizontal_alignment)

        # Horizontal concatenation
        if horizontal_alignment:
            new_image = Image.new('RGB', (curr_img.width + following_img.width, curr_img.height))
            new_image.paste(curr_img, (0, 0))
            new_image.paste(following_img, (curr_img.width, 0))

        # Vertical concatenation
        else:
            new_image = Image.new('RGB', (curr_img.width, curr_img.height + following_img.height))
            new_image.paste(curr_img, (0, 0))
            new_image.paste(following_img, (0, curr_img.height))
        return new_image


def plot_counterfactuals(smiles_list, experiments_paths_list, black_box_classifier, fig_save_path=None,
                         similarity_function="tanimoto_ecfp4", plot_subset=None, n_counterfactuals=3,
                         mol_size=(200, 200)):
    """
    Plotting the molecules to be explained along with their best scoring counterfactuals. Each row contains a target
    molecule along with its selected counterfactual explanations.
    :param smiles_list: SMILES list of the molecules that are being explained.
    :param experiments_paths_list: list of paths where each experiment is stored. Must match the length of smiles_list.
    :param black_box_classifier: xaimol.classifier.BlackBoxClassifier instance.
    :param fig_save_path: path where to save the PNG output. If None, the figure is not written on disk.
    :param similarity_function: keyword or instance of EvaluationStrategyComposant that estimates the distance between
    any point and the SMILES given in the constructor (default : "tanimoto_ecfp4").
    :param plot_subset: subset of experiments to be plotted (list of indices). If None, all experiments are plotted.
    :param n_counterfactuals: number of counterfactuals for each target.
    :param mol_size: (width, height) of each molecule in pixels
    :return: image
    """

    fig_smiles_list = []
    fig_class_list = []
    fig_class_prediction_list = []
    fig_similarity_list = []
    fig_target_img_list = []

    # Iterating over all target SMILES
    for i, target_smiles in enumerate(smiles_list):

        # Checking that current experiment must be plotted
        if plot_subset is None or i in plot_subset:

            # Computing distance function
            dist_fun = _get_distance_function(similarity_function, target_smiles)

            # Extracting best solutions in similarity order
            selected_solutions, _, _ = extract_counterfactuals_results(target_smiles, experiments_paths_list[i],
                                                                       black_box_classifier, similarity_function)
            selected_solutions = selected_solutions.tolist()

            # Iterating over all columns of current row
            for j in range(n_counterfactuals + 1):

                # If CF solution exists
                if j < len(selected_solutions):

                    curr_cf_smi = selected_solutions[j]

                    # Computing legend items for current molecule
                    fig_smiles_list.append(curr_cf_smi)
                    fig_class_list.append("+" if black_box_classifier.assess_class(curr_cf_smi) == "positive" else "-")
                    fig_class_prediction_list.append(
                        str("{:.2f}".format(black_box_classifier.assess_proba_value(curr_cf_smi))))
                    fig_similarity_list.append(str("{:.2f}".format(dist_fun.eval_smi(curr_cf_smi))))

                else:
                    # Default values if no CF exists
                    fig_smiles_list.append("H")
                    fig_class_list.append("")
                    fig_class_prediction_list.append("")
                    fig_similarity_list.append("")

            # Computing legend attributes for current target molecule
            target_mol_class = "+" if black_box_classifier.assess_class(target_smiles) == "positive" else "-"
            target_mol_prediction = str("{:.2f}".format(black_box_classifier.assess_proba_value(target_smiles)))
            target_mol_similarity = str("{:.2f}".format(dist_fun.eval_smi(target_smiles)))
            target_mol_legend = target_mol_class + ", " + target_mol_prediction + ", " + target_mol_similarity

            # Computing legend for current target molecule
            fig_target = plot_mol_counterfactuals_modification_map(target_smiles, selected_solutions, target_mol_legend,
                                                                   size=(mol_size[0], mol_size[1]),
                                                                   weights_threshold=0)

            # Resizing the image based on its definition and the input size parameters
            DPI = fig_target.get_dpi()
            fig_target.set_size_inches(mol_size[0] / float(DPI), mol_size[1] / float(DPI))

            # Converting figure to PIL image
            buf = io.BytesIO()
            fig_target.savefig(buf, dpi='figure', bbox_inches='tight')
            buf.seek(0)
            target_img = PIL.Image.open(buf)

            # Resizing the image to match the requested size in pixels due to the fact maplotlib does not guarantee
            # the final size (tight_layout issue).
            target_img = target_img.resize(mol_size, Image.ANTIALIAS)

            # Saving the image for current molecule in the list
            fig_target_img_list.append(target_img)

    # Computing list of legends for all explanations
    legends = [fig_class_list[i] + ", " + fig_class_prediction_list[i] + ", " + fig_similarity_list[i]
               for i in range(len(fig_smiles_list))]

    # Computing figure that contains all counterfactual explanations
    cf_explanations_img = MolsToGridImage([MolFromSmiles(s) for s in fig_smiles_list], legends=legends,
                                          molsPerRow=n_counterfactuals + 1, subImgSize=mol_size)

    # Vertical concatenation of all target images
    target_column_img = concatenate_images(fig_target_img_list, horizontal_alignment=False)

    # Concatenation of first column with the explanations
    img = concatenate_images([target_column_img, cf_explanations_img])

    # Writing figure to disk if necessary
    if fig_save_path is not None:
        with open(fig_save_path, "wb") as f:
            img.save(f, "png")

    # Returning output molecule
    return img
