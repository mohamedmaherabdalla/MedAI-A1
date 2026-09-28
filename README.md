ENG2440 Assignment 1 - RSNA pneumonia classification
Mohamed Abdalla (25011148)

Binary classification of chest X-rays from the RSNA Pneumonia Detection Challenge. The target is lung opacity consistent with possible pneumonia, not a confirmed pneumonia diagnosis. The negative class contains both Normal and No Lung Opacity / Not Normal exams.

What is in this repo
File	What it does
working.ipynb	The whole analysis, fully executed. Everything from loading the labels to Grad-CAM and the bonus questions is in here.
requirements.txt	The Python packages needed to run the notebook.
.gitignore	Keeps the data, the CSV files and the model weights out of Git.
No images, DICOM files, label/mapping CSVs or trained weights are in this repo. They have to be added locally (see below).

Notebook sections
Section	What it does
1. Setup	Picks the device (CUDA, Apple MPS or CPU) and imports.
2. Labels	Collapses the annotation rows to one label per exam (positive if any row has Target == 1).
3. NIH patients	Joins the RSNA-to-NIH mapping so the split can be done by original patient.
4. DICOM metadata	Reads the headers: image size, photometric interpretation, view (AP/PA), sex.
5. Images	Positive and negative examples, plus difficult cases with very small boxes.
6. Split	70/15/15 train/validation/test, grouped by NIH patient ID and stratified. Asserts that no patient is in two splits. The test set is not touched again until section 10.
7. Trivial baseline	Always predicts the majority class.
8. Simple baseline	Logistic regression on image intensity features.
9. Class weighting	Weighted vs unweighted loss on the simple baseline.
10. Baselines on test	Test-set results for the trivial and simple baselines.
11. Preprocessing	Modality LUT, inversion for MONOCHROME1, 1-99 percentile clipping, scaling to 0-1.
12. Augmentation	Small rotation and shift, training set only.
13. Data loaders	Dataset class and loaders (only train is shuffled).
14. Frozen ResNet18	Pretrained ResNet18 with only the final layer trained.
15. Weighted CNN	Same model with a class-weighted loss.
16. Fine-tuning	Fine-tunes layer4 + the final layer with early stopping on validation PR-AUC. Also an optional comparison with DenseNet121 and EfficientNet-B0.
17. Final model	Final ResNet18 trained with three seeds, plus the train vs validation loss curve.
18. Thresholds	Operating thresholds chosen on the validation set.
19. Test results	Confusion matrices, ROC, PR and calibration curves, and the final comparison table.
20. Subgroups	Results by view (AP vs PA) and by sex.
21. Grad-CAM	Grad-CAM on one TP, TN, FP and FN case.
22. Bonus	Bonus 1: prevalence shift and threshold transfer. Bonus 2: border masking with a lung-field control.
Setup
Tested with Python 3.12.


python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
Data (not included)
Put the files in this layout before running the notebook:


MedAI-A1/
|-- data/
|   |-- images/
|       |-- [extracted DICOM folders and files]
|-- working.ipynb
|-- assignment1_labels.csv
|-- rsna_to_nih_mapping.csv
The paths are set in section 2 of the notebook:


DATA_ROOT = Path("data/images")
LABEL_FILE = Path("assignment1_labels.csv")
MAPPING_FILE = Path("rsna_to_nih_mapping.csv")
Running
Open working.ipynb in Jupyter and run the cells from top to bottom. Some cells are slow: building the baseline features reads every DICOM (about 8 minutes), and the CNN training is much faster on a GPU.

The final models are saved to data/training_checkpoints/, which is inside data/, so they are not pushed to GitHub.

Seeds
What	Seed
Train/validation/test split	31
Example images in section 5	42
Frozen ResNet18, weighted CNN and backbone comparison (CNN_SEED)	42
Final ResNet18 runs	13, 31, 42
Bonus 1 prevalence subsampling	31
Grad-CAM and bonus 2	the seed 13 model
