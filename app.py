from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)

# Load pre-computed data (replace with your data loading logic)
def load_and_preprocess_data(csv_file, pickle_file):
    """
    Loads a CSV file, cleans and preprocesses the data, and stores it in a pickle file.

    Args:
        csv_file (str): Path to the CSV file containing job data.
        pickle_file (str): Path to the pickle file where preprocessed data will be stored.

    Returns:
        tuple: A tuple containing the preprocessed DataFrame, TF-IDF vectorizer, and TF-IDF matrix.
    """

    try:
        # Attempt to load data from pickle file
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
            df = data['df']
            tdif = data['tdif']
            tdif_matrix = data['tdif_matrix']
            print("Loaded data from pickle file")
            return df, tdif, tdif_matrix
    except FileNotFoundError:
        print("Pickle file not found, loading and preprocessing data from CSV...")

    # Load data from CSV if pickle file not found
    df = pd.read_csv(csv_file)
    df = df.dropna(subset=['jobdescription'])  # Remove rows with missing descriptions
    tdif = TfidfVectorizer(stop_words='english')
    tdif_matrix = tdif.fit_transform(df['jobdescription'])

    # Save preprocessed data for future use
    with open(pickle_file, 'wb') as f:
        data = {'df': df, 'tdif': tdif, 'tdif_matrix': tdif_matrix}
        pickle.dump(data, f)

    return df, tdif, tdif_matrix

csv_file = './saved_job_data_2.csv'
pickle_file = 'preprocessed_data.pkl'
df, tdif, tdif_matrix = load_and_preprocess_data(csv_file, pickle_file)

cosine_sim = linear_kernel(tdif_matrix, tdif_matrix)
indices = pd.Series(df.index, index=df['jobtitle']).drop_duplicates()

def get_recommendation(title, cosine_sim=cosine_sim):
    """
    Recommends jobs based on the provided job title using cosine similarity.

    Args:
        title (str): The job title for which to recommend similar jobs.
        cosine_sim (np.ndarray, optional): The pre-computed cosine similarity matrix. Defaults to cosine_sim.

    Returns:
        list: A list of recommended job titles (strings).
    """

    try:
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))

        if not sim_scores:
            print(f"No similar jobs found for '{title}'.")
            return []

        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        top_15_indices = [i for i, _ in sim_scores[:16]]  # Get top 15 (excluding input)

        recommendations = df['jobtitle'].iloc[top_15_indices[1:]]

        return recommendations.tolist()

    except KeyError:
        print(f"Job title '{title}' not found in the data.")
        return []


@app.route('/recommend', methods=['POST'])
def recommend_jobs():
    try:
        request_data = request.get_json()
        job_title = request_data['job_title']
    except KeyError:
        return jsonify({'error': 'Missing required field: job_title'}), 400

    recommended_jobs = get_recommendation(job_title)
    return jsonify(recommended_jobs)

if __name__ == '__main__':
    app.run(debug=True)
