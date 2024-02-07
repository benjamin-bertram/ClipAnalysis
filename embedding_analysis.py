import json
import numpy as np
import torch
import clip
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import pandas as pd

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

# Create a 3D scatter plot using Plotly
fig = go.Figure()

# Add text labels for the concepts
scatter = go.Scatter3d(
    x=df["x"],
    y=df["y"],
    z=df["z"],
    text=df["concept"],
    mode="text",
    textposition="middle center",
    showlegend=False,
)

fig.add_trace(scatter)

# Add lines connecting the points, with colors representing the distances
lines = []
# Define the range of colors for the lines based on distance
colors = ['blue', 'green', 'yellow', 'orange', 'red']
# Calculate distances and normalize them to get a value between 0 and 1
distances = []
for i in range(len(concepts)):
    for j in range(i + 1, len(concepts)):
        distance = np.sqrt((df.loc[i, "x"] - df.loc[j, "x"])**2 + 
                           (df.loc[i, "y"] - df.loc[j, "y"])**2 + 
                           (df.loc[i, "z"] - df.loc[j, "z"])**2)
        distances.append(distance)
max_distance = max(distances)
min_distance = min(distances)
normalized_distances = [(d - min_distance) / (max_distance - min_distance) for d in distances]

# Reset distances iterator for use in loop
distance_iter = iter(normalized_distances)

for i in range(len(concepts)):
    for j in range(i + 1, len(concepts)):
        # Use the next distance in the iterator
        normalized_distance = next(distance_iter)
        # Map the normalized distance to a color
        color_idx = int(normalized_distance * (len(colors) - 1))
        line_color = colors[color_idx]
        line = go.Scatter3d(
            x=[df.loc[i, "x"], df.loc[j, "x"]],
            y=[df.loc[i, "y"], df.loc[j, "y"]],
            z=[df.loc[i, "z"], df.loc[j, "z"]],
            mode="lines",
            line=dict(color=line_color, width=3),
            showlegend=False,
            hoverinfo="none",
            visible=False,
        )
        lines.append(line)
        fig.add_trace(line)

fig.update_layout(
    scene=dict(
        xaxis_title="Abstract-Concrete", #somewhat arbitrary interpretations
        yaxis_title="Detailed-Simple",
        zaxis_title="Pop-Classic",
    ),
    updatemenus=[
        dict(
            buttons=list([
                dict(
                    args=[{"visible": [True] + [False] * len(lines)}],
                    label="Show Distances",
                    method="update"
                ),
                dict(
                    args=[{"visible": [True] + [True] * len(lines)}],
                    label="Hide Distances",
                    method="update"
                )
            ]),
            direction="down",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.1,
            xanchor="left",
            y=1.1,
            yanchor="top"
        ),
    ],
    title="Concept Embeddings with Connection Distances",
    showlegend=False,
)

fig.show()
