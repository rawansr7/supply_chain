import openai
import pandas as pd
import json

# Set your OpenAI API key
with open("../data/openai_api_key.txt") as f:
    openai.api_key = f.read().strip()


# Function to generate embeddings
def get_embedding(text, model="text-embedding-3-large"):
    print("calling")
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
