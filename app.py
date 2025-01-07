from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load and preprocess the dataset
file_path = "Dataset.csv"
df = pd.read_csv(file_path)
df = df.dropna(subset=['Delayed By', 'Price', 'Turn Around Time', 'Duration'])

# Prepare the data for clustering
X = df[['Delayed By', 'Price', 'Turn Around Time', 'Duration']].values

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply DBSCAN
dbscan = DBSCAN(eps=1.0, min_samples=10)
dbscan_labels = dbscan.fit_predict(X_scaled)

df['Cluster'] = dbscan_labels

# Compute cluster descriptions
cluster_descriptions = df.groupby('Cluster')[['Delayed By', 'Price', 'Turn Around Time', 'Duration']].mean()

# Flask app setup
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', cluster=None, description=None, message=None, error=None)


@app.route('/get-cluster-description', methods=['POST'])
def get_cluster_description():
    try:
        # Parse the input data
        input_data = request.form
        data_point = [
            float(input_data.get('Delayed By')),
            float(input_data.get('Price')),
            float(input_data.get('Turn Around Time')),
            float(input_data.get('Duration'))
        ]

        # Scale the input data
        data_point_scaled = scaler.transform([data_point])

        # Predict the cluster
        cluster_label = dbscan.fit_predict(np.vstack([X_scaled, data_point_scaled]))[-1]

        if cluster_label == -1:
            return render_template('index.html', message="The data point belongs to noise.")

        # Get the cluster description
        cluster_description = cluster_descriptions.loc[cluster_label].to_dict()

        return render_template('index.html', cluster=int(cluster_label), description=cluster_description)
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
