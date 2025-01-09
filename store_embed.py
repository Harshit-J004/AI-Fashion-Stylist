import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from PIL import Image
import os

# Resize images before processing
def preprocess_image(uri, size=(512, 512)):
    try:
        img = Image.open(uri)
        img = img.resize(size)
        img.save(uri)  # Save the resized image
    except Exception as e:
        print(f"Error resizing image {uri}: {e}")

# Set up database
dataset_folder = 'Data'
chroma_client = chromadb.PersistentClient(path="Vector_database")
image_loader = ImageLoader()
CLIP = OpenCLIPEmbeddingFunction()

image_vdb = chroma_client.get_or_create_collection(name="image", embedding_function=CLIP, data_loader=image_loader)

# Prepare image paths and IDs
ids = []
uris = []

for i, filename in enumerate(sorted(os.listdir(dataset_folder))):
    if filename.endswith('.png'):
        file_path = os.path.join(dataset_folder, filename)
        preprocess_image(file_path)  # Resize image before adding
        ids.append(str(i))
        uris.append(file_path)

# Process images in batches
batch_size = 10
for i in range(0, len(uris), batch_size):
    batch_ids = ids[i:i+batch_size]
    batch_uris = uris[i:i+batch_size]
    try:
        image_vdb.add(
            ids=batch_ids,
            uris=batch_uris
        )
    except Exception as e:
        print(f"Error processing batch {i}: {e}")

print("Images stored to the Vector database.")
