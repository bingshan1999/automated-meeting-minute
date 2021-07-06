# automated-meeting-minute
This project introduces 2 methods to produce a meeting minute from a plain transcript.

# Method 1: Classifying important sentences using Convolutional Neural Network
This method extracts important sentences out of the long and unstructured transcript. It uses NLP techniques to preprocess the transcript and convert it into matrix representation using GloVe embedding. CNN is then used to to extract n-gram features at different position of a sentence thruogh a 1D convolutional filter.
Note: The data used in this project is from AMI corpus https://groups.inf.ed.ac.uk/ami/corpus/

### The code can be run in Google Collaboraroty (.ipynb file)
1) Import all the file in data/ into Collab.
2) Obtain pretrained Wikipedia GloVe embedding https://nlp.stanford.edu/projects/glove/ and import them into Collab.
3) Run the code.

# Method 2: Lexical-based extraction to produce a meeting minute
This method uses concept of repetitive noun to produce segments of important topics in a meeting transcript. Noise filtering e.g., semantic similarity is used to filter out stopwords and less relevant topics to improve the quality of minute produced.

### This code can be run in python terminal.
1) Configure the settings file to adjust parameters and file paths.
2) Run fyp.py and a minute file will be generated.
