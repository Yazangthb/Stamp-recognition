# Stamp recognition

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Description

Our project offers a solution for stamp detection, classification, and comparison using Maching learning and Computer
Vision techniques. The primary goal is to facilitate the identification and analysis of stamps present in document
scans.

## Authors and acknowledgment

- **Sirojiddin Komolov: Author, Client**
- Sofia Tkachenko: Team Lead, Backend developer
- Osama Orabi: ML engineer, datascience
- Yazan Alnakri: Computer vision
- Leonid Pustobaev: Dataset Augmentation
- Laith Nayal: Computer vision
- Ahmed Abid: ML engineer

## Demo
### Classify the stamp on the image
- ![Request](https://drive.google.com/file/d/11BNO1TdvKf4hVhaczPbF38tG8mK1rkmC/view?usp=drive_link)
- ![Good response](https://drive.google.com/file/d/1gidMT8Ohe04btB_s1zTkC3ynDJZ4kwL3/view?usp=drive_link)
- ![Response with errors:](https://drive.google.com/file/d/1iBvDWP606L1fo8l4DhSpGclE4i_1qanc/view?usp=drive_link)
### Add new stamp to the database
- ![Request](https://drive.google.com/file/d/15gpQR2Ytj8OPzcApF_1U6TOre-UZe4wG/view?usp=drive_link)
- ![Response](https://drive.google.com/file/d/1Z8gMqiPFSar0Cf3yB8PTDBy7X3GEwZdz/view?usp=drive_link)

## How to use

- POST /images/upload - allows the user to upload images that they want to detect and classify the stamps on.
    - Input - scans of document in an image format (png, jpg) and the names for the corresponding stamps.
    - Output - [{stamp_name, stamp_picture, accuracy}, …]
- POST /images/add_stamp - allows the user to add new stamps to the database.
    - Input - {[stamp_name1, stamp_name2, ...], document_picture}
- Requests that will be added soon
    - POST /images/compare - allows the user to compare two images of the stamps in order to check validity.
        - Input - 2 scans of documents in an image format.
        - Output - similarity score.
    - GET /help - returns instructions on how to use the API.

## What files are where?

- API module:
    - myapp.py - main file of the backend
    - database.py - supporting functions to work with the database
- Detection module:
    - Detect_grayscale_stamps.py - old detection by blurring and contour
    - Detection_Model.ipynb - new detection models' notebook (but all the models are loaded from Roboflow server)
- Embeddings (some of the embedding experiments):
    - new_CNN_for_embeddings.ipynb - the notebook for the latest model we use
- test:
    - api.py - unittests
    - images - folder of test images

## Features

Our project offers the following key features, divided into four parts:

### 1. Data Augmentation

- **Creation of a dataset**: We generate a comprehensive dataset of documents with stamps by combining real documents
  with stamps generated with Stable Diffusion.

### 2. Stamp Detection

- **Detection**: Our system identifies stamp(s) present in document images using several custom CNN models.
- **Location determination**: We detect and precisely locate a frame on the detected stamp(s) on the document, providing
  their coordinates.

### 3. Stamp Embedding

- **Feature extraction**: We utilize our own CNN model to generate embeddings from stamp images.
- **Vectorization of stamps**: The embeddings represent stamps as high-dimensional vectors, capturing their unique
  characteristics.

### 4. Classification

- **Distance calculation**: From the embeddings, we calculate the cosine similarity between the embedding of the current
  stamp and all embeddings in the SQLite database.
- **Classification**: Based on the calculated distances, our system provides reliable classification results for stamps.

## Technologies used

- Backend: Python, SQLight, Flask, Pydantic, Werkzeug, unit-testing
- Detection: Edge detection, Blurring, Python, cv2, matplotlib, numpy
- Embeddings: CNN, Image preprocessing, Python, tensorflow, numpy, scipy, matplotlib
- Dataset augmentation: Stable diffusion, Photoshop, Blender, Python

## Limitations

- _Backend_ - have yet to implement document to image conversion due to the higher priority of detection improving task.
  If this or two images comparison is needed, contact @DablSi in telegram and I will finish it in a day.
- _Detection_ - since this model was our latest experiment and was only complete as of 19/07/2023, we have only trained
  it on circular and square stamps. The performance on such stamps is great - 99% accuracy on test and real data, but
  hand markup of data takes a lot of time. If the model will be used, we can markup 60k stamps more to include all
  possible types.
- _Embedding_ - even though real data accuracy is 80-85% after detection, we still would like to improve that be
  additional training, since we have several more ideas on how to improve the result.