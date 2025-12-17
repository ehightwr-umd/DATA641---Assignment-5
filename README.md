# DATA641 - Assignment 5

Course (PCS): DATA 641 (PCS4)
Author: Emily Hightower
Date Due: 2025.12.16

The following project and text are responses adapted from "Homework 5" by Professor Naeemul Hassan (2025, UMD).
----------

# Comparative Analysis of Gun Violence Coverage Across News Outlets

## Project Description:
This project applies different Natural Language Processing (NLP) techniques to analyze how different news outlets portray victims and shooters in gun violence incidents. Using a provided dataset of 100 articles from four sources, the project performs:
- Content Extraction Using Coreference Resolution
- Description or Phrase Extraction
- Embedding-Based Clustering of Descriptions
- Statistical Analysis
The resulting analysis explores how descriptive language is used to characterize entities in news coverage about gun violence. 

This document will describe the dataset and pre-processing, explain each section of the code, and provide a brief analysis of the results. The code is designed to look for existing files, and generate new files if missing.

## Dataset:
The dataset contains approximately 100 articles from the news outlets: CNN, Fox News, New York Times (NYT), and Wall Street Journal (WSJ). Each journal offers twenty-five example articles, mostly from the month of May with approximately 5 paragraphs per article. Before use in natural language processing, the data needs to be combined, organized, and cleaned.

**Dataset Link:** https://drive.google.com/drive/folders/1R3UN1BktHtvnZu3n9IXsizZOeaSuCh-i?usp=share_link

## Program Components & Documentation

### Phase 0: Data Processing
This phase processes the data and performs a sentence-level extraction pipeline for the articles. The code combines the unique files from the dataset into a single CSV file. It aims to standardize the news text, preserve metadata (outlet, article context, date, sentence number), and creates a dataframe for the next phases. 

### Phase 1: Context Extraction & Labelling
This section aims to identify all text segments that reference the victim(s) and shooter(s) in each article and attempt to resolve coreferences to create coherent contexts. Articles are processed as sentences, which unfortunately reduces preservation of long-range references across sentences. The articles are segmented into sentences using spaCy, and the sentences are classified as referencing victims or shooters based on identified lexicons. The code implemented in this phase generates two datasets, one focused on victims and one on shooters.

A key challenge in this phase was compatability. Most coreference resolution tools are not compatible in Google Colab, requiring a switch to GitHub and Visual Studio Code. Within VSC, AllenNLP and neuralcoref are incompatible with current python versions, requiring a Hugging Face model. The Hugging Face model required API token access. Ultimately the coreference resolution piece model was removed due to the author's inability to work with any of the provided models (neuralcoref, AllenNLP, Hugging Face coref, fastcoref, etc.). Additional limitations include the inability of anonymized references and rule-based sentences to capture subtle or implicit references.

### Phase 2: Description Extraction
In the next phase of the pipeline, the coreference resolution CSV files are processed to identify descriptive words, phrases, and expressions used to characterize victims and shooters from gun violence news coverage. The approach combines the following approaches: rule-based linguistic extraction, dependency parsing, and sentiment analysis. Using spaCy's dependency parser, the code identifies descriptive parts-of-speech phrases associated with victims and shooters. Additional regular expressions, for example common numeric casualty patterns, are extracted Sentence-level sentiment polarity is computed using TextBlob to capture the emotion around language surrounding violence events.

The limitations of this section are as follows:
- Common convention in journalism anonyomizes shooters, limiting specificity of descritions.
- Rule-based heuristics are limited, and may miss implicit, metaphorial, or contextually nuanced descriptions.
- Sentiment analysis is applied at the sentence-level and does not always distinguish sentiment for sentences containing references to both shooters and victims.

### Phase 3: Description Clustering
This code processes the descriptions from Phase 2 to create semantically similar descriptions as clusters, identifying patterns in how the victim and shooter entities are portrayed. Generally, the code preproceses the descriptions by splitting, cleaning, deduplicating and lower casing the information. It then embeds the description using Sentence-BERT (Option C), then reduces dimensionality using UMAP before clustering. Clustering is performed using HDBSCAN which automatically identifies clusters and noise without a fixed number of clusters. The code merges the clusters based on cosine similarity of centroids and saves the clustered data for analysis.

In this section, the hyperparameters are min_cluster_size, set to 3, and a tunable cluster_selection_epsilon, set to 0.05. For the first few attempts, clusters were too large with broad hyperparameters. Tuning these parameters allowed more diverse clustering to occur, leading to the output received. Visualizations of the UMAP analysis are included in the visualization.ipynb document.

### Phase 4: Manual Cluster Evaluation
After completing the first four phases, the fifth phase used a manual cluster evaluation to determine effectiveness. The results were very broad, with similar but slightly different lexical versions dominating the clustering. Semantically, many clusters were similar with many connecting to the number of victims, however some were identifiers like "massacre", "child", "teacher", or "grandmother". Manual adjustments were considered to reduce the number of total clusters (19 for victims). This would be completed by combining lexically and semantically similar clusters, for example "19 children" and "at least 19 children". Many clusters were pure, but a few instances of noise occured, for example, when NYT reported the total number of school shootings in a year. Ultimately, clusters were not altered to better understand how the initial programming results performed.

One difficulty from the Phase 3 and 4 was the lack of cluster labeling beyond numeric values. Future implementation would hopefully improve this process.

Victim Cluster Labels include:
- Cluster -1: Noise
- Cluster 0: "at least 19 children"
- Cluster 1: "# children"
- Cluster 2: "shooting"
- Cluster 3: "massacre"
- Cluster 5: "students"
- Cluster 6: "two teachers"

Shooter Cluster Labels include:
- Cluster 0: "alleged shooter"
- Cluster 1: "shooter"
- Cluster 2: "gunman"
- Cluster 3: "suspect"

### Phase 5: Cross-Outlet Frequency Analysis
Using the original cluster analysis, this phase created comprehensive tables showing how frequently each cluster appears in the news outlets and entity types. Both proportion and frequency tables are available in the processed files. Visualizations of clusters, heatmaps, and grouped bar charts are similarly available in figures.

### Phase 6: Statistical Hypothesis Testing
In this phase, statistical testing is performed on the top three most frequent clusters from the different entities: victims and shooters. The results from this are saved in the file "6_hypothesis_testing.md". Visualizations of the Chi-Squared Test of Homogeneity are availabile in figures.

---------------

## Code Implementation:
- pip install -r requirements.txt
- python main_pipeline.py


## Citations

Hassan, N. "Homework 5". University of Maryland, ELMS. November 2025, https://docs.google.com/document/d/1FQ5KzFA-ubKp4QoGO_ISQ24f055vUisn_dr7EH4PNvM/edit?usp=sharing.