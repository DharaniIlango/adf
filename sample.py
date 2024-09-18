from huggingface_hub import snapshot_download
dataset_path = snapshot_download(repo_id="LivingOptics/hyperspectral-fruit", repo_type="dataset")
print(dataset_path)

import os.path as op
import numpy.typing as npt
from typing import List, Dict, Generator
from lo.data.tools import Annotation, LODataItem, LOJSONDataset, draw_annotations
from lo.data.dataset_visualisation import get_object_spectra, plot_labelled_spectra
from lo.sdk.api.acquisition.io.open import open as lo_open

# Load the dataset
path_to_download = op.expanduser("~/Downloads/hyperspectral-fruit")
dataset = LOJSONDataset(path_to_download)

# Get the training data as an iterator 
training_data: List[LODataItem] = dataset.load("train")

# Inspect the data
lo_data_item: LODataItem
for lo_data_item in training_data[:3]:

    draw_annotations(lo_data_item)

    ann: Annotation
    for ann in lo_data_item.annotations:
        print(ann.class_name, ann.category, ann.subcategories)

# Plot the spectra for each class
fig, ax = plt.subplots(1)
object_spectra_dict = {}
class_numbers_to_labels = {0: "background_class"}
for lo_data_item in training_data:
    object_spectra_dict, class_numbers_to_labels = get_object_spectra(
        lo_data_item, object_spectra_dict, class_numbers_to_labels
    )

plot_labelled_spectra(object_spectra_dict, class_numbers_to_labels, ax)
plt.show()
