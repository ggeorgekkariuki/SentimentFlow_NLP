import pickle

# Load your pre-trained model from a pickle file
with open('app/models/tuned_lr_tf_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the vectorizer
with open('app/models/vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Dataset
with open('app/models/evaluation_df.pkl', 'rb') as file:
    data = pickle.load(file)

# Dataset
with open('app/models/labels.pkl', 'rb') as file:
    label_categories = pickle.load(file)