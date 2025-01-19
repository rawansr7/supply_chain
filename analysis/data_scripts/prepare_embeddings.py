import openai
import pandas as pd
import json

# Set your OpenAI API key
openai.api_key = "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"


# Function to generate embeddings
def get_embedding(text, model="text-embedding-3-large"):
    response = openai.Embedding.create(input=text, model=model, dimensions=256)
    return response["data"][0]["embedding"]


df = pd.read_csv("../data/labelled_products.csv")

# Create a dictionary with item_id as keys and embeddings as values
embeddings_dict = {}
for _, row in df.iterrows():
    item_id = row["item_id"]
    description = row["product_description"].strip()
    embedding = get_embedding(description)
    embeddings_dict[item_id] = embedding

# Save the embeddings dictionary to a JSON file
output_file = "../data/embeddings.json"
with open(output_file, "w") as f:
    json.dump(embeddings_dict, f)
