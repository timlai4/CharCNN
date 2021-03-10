# CharCNN
CNN model to detect handwritten characters

## Dataset
We used the Kaggle dataset https://www.kaggle.com/sachinpatel21/az-handwritten-alphabets-in-csv-format which combined data from various sources including the NIST and MNIST datasets.

The data came in the format of a single CSV file, with the first column denoting the labels and the rest of 784 = 28 x 28 columns representing the images. We followed the guide in https://data-flair.training/blogs/handwritten-character-recognition-neural-network/ to build the input pipeline and then used https://github.com/timlai4/CharCNN/blob/main/parse_df.py to subset the data for the specific characters (ACTG) that we wanted and split into training and cross-validation.
