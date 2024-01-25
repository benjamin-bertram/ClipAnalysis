from dotenv import load_dotenv
# Load default environment variables (.env)
load_dotenv()

import os
import json
import numpy as np
import torch
import clip
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import pandas as pd
import openai
import requests


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load the concepts from the file
with open("concepts.txt", "r") as f:
    concepts = [line.strip() for line in f.readlines()]

def get_embedding(text):
    with torch.no_grad():
        inputs = clip.tokenize([text]).to(device)
        outputs = model.encode_text(inputs)
    return outputs.cpu().numpy()

embeddings = [get_embedding(concept) for concept in concepts]

embeddings = np.array(embeddings).reshape(len(concepts), -1)

embeddings_dict = {concept: embedding.tolist() for concept, embedding in zip(concepts, embeddings)}
with open('embeddings.json', 'w') as f:
    json.dump(embeddings_dict, f)

pca = PCA(n_components=3)
pca.fit(embeddings)
embeddings_3d = pca.transform(embeddings)

# Create a DataFrame with the coordinates
df = pd.DataFrame(embeddings_3d, columns=["x", "y", "z"])
df["concept"] = concepts

# Apply K-means clustering
kmeans = KMeans(n_clusters=8, random_state=0).fit(embeddings)
df["cluster"] = kmeans.labels_

# Function to characterize a cluster
def characterize_cluster(cluster_df):
    characteristics = {}
    characteristics['mean'] = cluster_df[['x', 'y', 'z']].mean().to_dict()
    characteristics['std'] = cluster_df[['x', 'y', 'z']].std().to_dict()
    return characteristics

def generate_cluster_name(concepts):
    # Format the concepts into a string
    concepts_str = ', '.join(concepts)
    print(concepts_str)

    # Prompt for ChatGPT
    prompt = f"Please generate a short name that describes the common theme of these concepts: {concepts_str}"

    # Initialize the OpenAI API
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    #openai.api_key = OPENAI_API_KEY

    # Call the OpenAI API
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
            {"role": "system", "content": "You are a highly reknown art historian."},
            {"role": "user", "content": prompt},
        ],
    )
    
    # Get the assistant's message from the response
    assistant_message = response['choices'][0]['message']['content']

    return assistant_message.strip()

def get_image_url(artist_name):
    # Replace spaces in the artist name with '+'
    artist_name = artist_name.replace(' ', '+')

    # Construct the URL for the API request
    url = f"https://commons.wikimedia.org/w/api.php?action=query&format=json&list=search&srsearch={artist_name}&utf8=1&formatversion=2"

    # Send the API request
    response = requests.get(url)

    # Parse the JSON response
    data = response.json()

    # Check if the search returned any results
    if data['query']['searchinfo']['totalhits'] > 0:
        # Get the title of the first result
        title = data['query']['search'][0]['title']

        # Construct the URL for the image
        image_url = f"https://commons.wikimedia.org/wiki/File:{title}"

        return image_url
    else:
        # Return a URL for a black square image
        return "https://via.placeholder.com/200.png/000000/000000"

cluster_names = []

for cluster_label in df['cluster'].unique():
    # Get the data for this cluster
    cluster_df = df[df['cluster'] == cluster_label]
    
    # Get the list of concepts in this cluster
    concepts = df['concept'].unique()
    
    # Generate a name for the cluster
    cluster_name = generate_cluster_name(concepts)
    
    cluster_names.append(cluster_name)

    # Get an image URL for each concept
    image_urls = {concept: get_image_url(concept) for concept in concepts}

    # Add the image URLs to the DataFrame
    df['image'] = df['concept'].map(image_urls)

# Map cluster labels to names
cluster_name_map = {label: name for label, name in enumerate(cluster_names)}

# Add a new column to the DataFrame with the name of each point's cluster
df['cluster_name'] = df['cluster'].map(cluster_name_map)

# Generate names for the axes
x_axis_concepts = df.nsmallest(5, 'x')['concept'].tolist() + df.nlargest(5, 'x')['concept'].tolist()
y_axis_concepts = df.nsmallest(5, 'y')['concept'].tolist() + df.nlargest(5, 'y')['concept'].tolist()
z_axis_concepts = df.nsmallest(5, 'z')['concept'].tolist() + df.nlargest(5, 'z')['concept'].tolist()

x_axis_name = generate_cluster_name(x_axis_concepts)
y_axis_name = generate_cluster_name(y_axis_concepts)
z_axis_name = generate_cluster_name(z_axis_concepts)

# Create a scatter plot of the embeddings
scatter = go.Scatter3d(
    x=df['x'],
    y=df['y'],
    z=df['z'],
    mode='markers+text',  # Add 'text' to the mode to display text labels
    text=df['concept'],  # This will be the text displayed next to each point
    hovertext=df[['concept', 'cluster_name', 'image']].apply(lambda x: f"<img src='{x[2]}' width='200' height='200'><br>{x[0]}<br>{x[1]}", axis=1),
    hoverinfo='text',
    textposition='top center',  # Change this to position the text labels as needed
    marker=dict(
        size=8,
        color=df['cluster'],  # set color to the cluster labels
        colorscale='Viridis',  # choose a colorscale
        opacity=0.8
    )
)

layout = go.Layout(
    scene=dict(
        xaxis_title=x_axis_name,
        yaxis_title=y_axis_name,
        zaxis_title=z_axis_name
    ),
    margin=dict(l=0, r=0, b=0, t=0)
)

fig = go.Figure(data=[scatter], layout=layout)

# Function to update the scatter plot's visibility based on the clicked point
def update_point(trace, points, selector):
    # Get the cluster of the clicked point
    clicked_point_cluster = df.iloc[points.point_inds[0]]['cluster']

    # Identify points that are in the same cluster
    same_cluster_points = df['cluster'] == clicked_point_cluster
    
    # Create a new opacity array: 0 for points in other clusters, 0.8 for points in the same cluster
    new_opacity = np.where(same_cluster_points, 0.8, 0)
    
    # Update the marker opacity
    scatter.marker.opacity = new_opacity
    fig.update_traces(marker=scatter.marker)


# Bind the function to the scatter plot's click event
scatter.on_click(update_point)

fig.show()
