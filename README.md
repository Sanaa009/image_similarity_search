
# Image Captioning and Similarity Search Using CLIP, FAISS, and Vision-Encoder-Decoder Models

This project demonstrates how to use **CLIP**, **Vision-Encoder-Decoder models** for image captioning, and **FAISS** for vector similarity search. It enables the user to:
- Generate image captions based on uploaded images.
- Find similar images using vector embeddings stored in a FAISS index.

## Features
- **Image Captioning**: Generates captions for images using a pre-trained `ViT-GPT2` model.
- **Image Embeddings**: Extracts image features using the `CLIP` model.
- **Similarity Search**: Finds similar images using FAISS for vector search.

## Models Used
1. **CLIP** (`openai/clip-vit-base-patch32`): For generating image embeddings.
2. **Vision-Encoder-Decoder** (`nlpconnect/vit-gpt2-image-captioning`): For generating captions from image data.

## Requirements

To run this project, the following Python libraries are required:

- `transformers`
- `torch`
- `torchvision`
- `Pillow`
- `faiss-cpu`
- `huggingface_hub`
- `google.colab` (for uploading files in Colab)

You can install the dependencies using the following commands:

```bash
pip install transformers torch torchvision Pillow faiss-cpu huggingface_hub
```

## Running the Notebook

1. **Upload Images**: Upload at least 5 images for the model to process.
2. **Generate Captions**: The script will generate captions for each uploaded image.
3. **Similarity Search**: You can upload a new image and the model will find the most similar images from the uploaded dataset using FAISS.

## How It Works

- **Step 1**: The CLIP model is used to generate embeddings for the uploaded images.
- **Step 2**: The `ViT-GPT2` model is used to generate captions for the images.
- **Step 3**: All the image embeddings are indexed using FAISS, and when a new image is uploaded, its embedding is compared with those in the FAISS index to find similar images.

## Code Walkthrough

### Import Libraries and Initialize Models

- Import necessary libraries: `torch`, `transformers`, `Pillow`, `faiss`, `matplotlib`, etc.
- Load the CLIP and Vision-Encoder-Decoder models from Hugging Face.

### Uploading Images

- The user uploads images through the interface.
- Each uploaded image is processed to extract embeddings and generate captions.

### Image Captioning

- The image is processed using the `ViT-GPT2` captioning model to generate human-readable captions.

### Image Similarity Search

- The CLIP model is used to generate image embeddings, which are stored in a FAISS index.
- For a query image, its embedding is compared with stored embeddings, and the most similar images are retrieved.




