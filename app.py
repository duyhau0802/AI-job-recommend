from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from flask_cors import CORS
import json
import mysql.connector
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

csv_file = './job_data.csv'
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

def save_recommendation():
    try:
        # Connect to the MySQL database
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="tuyendung",
        )

        # Create a cursor object
        cursor = connection.cursor()

        # Define the SQL query to select data from your table
        sql_query = "SELECT `vi_tri`, `mo_ta`, `address_cong_viec` FROM `jobs`" 

        # Execute the query
        cursor.execute(sql_query)

        # Fetch all results as a list of tuples
        data = cursor.fetchall()

        # # Create a DataFrame from the fetched data
        dfDB = pd.DataFrame(data, columns=[col[0] for col in cursor.description], index=None)  # Extract column names
        dfDB.rename(columns={'vi_tri': 'jobTitle', 'mo_ta': 'jobdescription', 'address_cong_viec': 'joblocation_address'}, inplace=True)
        dfDB.to_csv('dataStore.csv', index=True, header=True)
        return True
    except mysql.connector.Error as err:
        print(f"Error connecting to database: {err}")
        return False
    finally:
        if connection:
            connection.close()
            cursor.close()

@app.route('/recommend', methods=['POST'])
def recommend_jobs():
    try:
        request_data = request.get_json()
        job_title = request_data['job_title']
    except KeyError:
        return jsonify({'error': 'Missing required field: job_title'}), 400

    recommended_jobs = get_recommendation(job_title)
    return jsonify(recommended_jobs)

@app.route('/saveDataRecommend', methods=['GET'])
def saveDataRecommend():
    saveDataSuccess = save_recommendation()
    if saveDataSuccess:
        return jsonify({'message': 'Data saved successfully'})
    else:
        return jsonify({'message': 'Failed to save data'})
if __name__ == '__main__':
    app.run(debug=True)
