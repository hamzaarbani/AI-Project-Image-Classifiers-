# ML Project Lifecycle – Development & Deployment Phase

## Project Overview

This project demonstrates the complete Machine Learning lifecycle including model development, API creation, containerization, and deployment. The objective of this project is to understand the Development and Deployment phases of an ML system using a pretrained deep learning model, Flask API, and Docker.

The project includes model training, batch inference, evaluation metrics, API development using Flask, and deployment using Docker Hub and GitHub.

---

# Objectives

The main objectives of this project are to understand the Development and Deployment phases of the Machine Learning lifecycle, learn the constraints and tasks involved in both phases, convert ML scripts into reusable Python files, deploy ML models using Flask APIs, and containerize applications using Docker.

---

# Project Structure

```plaintext
Project/
│
├── Dataset/
│
├── Notebook/
│   └── Model_Training.ipynb
│
├── Train.py
├── Inference.py
├── app.py
├── requirements.txt
├── Dockerfile
├── README.md
│
├── Saved_Model/
│   └── model.pt
│
└── Results/
    └── evaluation_results.txt
```

---

# Development Phase

## Task 1 – Model Training and Evaluation

A pretrained deep learning model was fine-tuned using the provided dataset. The dataset was used for training, validation, and testing purposes.

Dataset Link:
[Google Drive Dataset](https://drive.google.com/file/d/1kugP9qyUQ4FZ4T8HWNqhUCpQ-cnkaHxf/view?usp=sharing&utm_source=chatgpt.com)

The project includes:

* Fine-tuning a pretrained model
* Model evaluation
* Batch inference for 4 samples
* Reporting evaluation metrics

### Evaluation Metrics

The following metrics were used to evaluate the model performance:

* Accuracy
* Precision
* Recall
* F1-Score

Example:

```plaintext
Accuracy  : 92%
Precision : 91%
Recall    : 90%
F1-Score  : 90.5%
```

---

## Task 2 – Python Scripts and Flask API

The notebook code was converted into separate Python files:

### Train.py

This file is responsible for:

* Loading dataset
* Preprocessing data
* Training the model
* Saving trained weights

### Inference.py

This file performs:

* Model loading
* Prediction on input samples
* Batch inference

### app.py (Flask API)

The inference system was converted into a Flask API for deployment purposes.

The API accepts requests and returns model predictions in JSON format.

Example API Endpoint:

```plaintext
POST /predict
```

Example Response:

```json
{
  "prediction": "Positive"
}
```

---

# Deployment Phase

## Dockerization

The Flask API was containerized using Docker.

### Dockerfile

A Dockerfile was created to:

* Define the application environment
* Install dependencies
* Copy project files
* Run Flask API automatically

Example Commands:

```bash
docker build -t ml-project .
docker run -p 5000:5000 ml-project
```

---

## Docker Hub

The Docker image was uploaded publicly on Docker Hub.

Docker Hub Link:

```plaintext
Add your Docker Hub link here
```

---

## GitHub Repository

The complete source code of the project is uploaded on GitHub.

GitHub Repository Link:

```plaintext
Add your GitHub repository link here
```

---

# Documentation Phase

## What is API?

An API (Application Programming Interface) is a system that allows communication between different software applications. It enables applications to exchange data and functionalities through requests and responses.

---

## Difference between Flask and FastAPI

Flask is a lightweight web framework mainly used for building web applications and APIs. FastAPI is a modern high-performance framework specifically designed for APIs with built-in asynchronous support and automatic validation.

---

## What is REST Framework for APIs?

REST Framework is a standard architecture used to develop APIs using HTTP methods such as GET, POST, PUT, and DELETE. It enables communication between client and server systems in a structured way.

---

## What are Microservices and how ML developers use them?

Microservices are small independent services that work together within a larger application. ML developers use microservices
