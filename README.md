# check-worthiness

## Name
Check Worthiness: Checking Whether a Tweet-Image Pair Requires Fact Checking!

## Streamlit UI Usage 
Run the following command in the main directory of the repository using the correct virtual environment for the project. 

⚠️ Make sure to fetch the 1.48 GBGB imgs_gold.pickle file from the project storage on Google Drive. This file was excluded to keep the repository free from large files.

Link to imgs_gold.pickle (request access if necessary): https://drive.google.com/file/d/1DJK44D_UW_u2s_VzLBLqlIuQSdEhtvOw/view?usp=drive_link

`python -m streamlit run ui/main.py`


## Description
Multimodal classification task on check-worthiness of tweet-image pairs collected from Twitter. The task is part of the "Check That! 2023" competition which in turn is part of CLEF 2023: https://checkthat.gitlab.io/clef2023/

## Visuals
To-be-added.

## Installation
Install the provided requirements.txt in a Python virtual environment preferably with Python version 3.10.9. Create a Jupyter Kernel for this virtual environment to be use the pipelines and the analysis notebooks.

## Usage
### UI 
- Run the UI and display it on a web browser (port on local host provided on terminal after running the Streamlit command)
### Notebooks 
- Use pipeline notebooks to extract text and/or image embeddings from tweet-image pairs, train neural networks using our base datasets, dataloaders, models scripts, fit an SVM, or prompt ChatGPT to do classification. 
- Use analysis notebooks to examine model output scores, feature distributions, PCA on features, etc. 

## Authors and Acknowledgment
Onur Deniz Güler deniz.gueler@tum.de        
Jonas Engesser j.engesser@tum.de            
Patrizio Palmisano patrizio.palmisano@tum.de        
