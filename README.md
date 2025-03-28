# Veridion Project - Company Classifier

## Description
This project serves as an application for the DeepTech Engineer Intern role at Veridion. The goal of the project is to build a robust company classifier based on a new insurance taxonomy. The classifier uses natural language processing (NLP) techniques and machine learning models to categorize companies into different sectors, categories, and niches, based on their descriptions and business tags.

## Datasets

The project uses two datasets:
1. **Companies Dataset**: Contains information about companies, such as their description, business tags, sector, category, and niche.
2. **Insurance Taxonomy Dataset**: Contains a taxonomy of insurance labels used for categorizing companies.

## Steps Taken

### Data Analysis
- Checked for null values in the datasets. Since the number of null values was very small, I dropped the rows containing them.
- Split the business tags into different columns to explore relationships with the sectors.
- Analyzed the connection between company categories and the taxonomy labels, and found minimal overlap.

### Formatting Data for FastText
- Preprocessed the data for FastText by converting the business tags into labels and concatenating the relevant company information into a text column (`final_text`) to be used for training the model.

### Splitting the Dataset
- Split the dataset into 80% training data and 20% test data, with stratification based on company sectors. The training set was further divided into 90% for training and 10% for validation.
- Saved only the `final_text` column for the training and validation sets in `.txt` format to be used in FastText training.
- The test set retained the original columns with an additional text column combining all relevant company information.

### Model Training
- Trained FastText models with different parameters and evaluated their performance based on accuracy and recall.
- The best-performing model achieved an accuracy of 90.49%, but its recall was low at 32.11%, indicating challenges with classifying less frequent labels.

### Companies and Taxonomy Comparison
- Used the top 10 most common words in `final_text` to create word vectors.
- Compared these vectors with the vectors in the `label_norm` column of the taxonomy to determine similarities.
- Assigned the `insurance_label` based on the highest number of similarities.


