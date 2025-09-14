# 🌿 Plant Disease Detection

This project provides a deep learning-based solution to detect and classify diseases in plant leaves. The core of the system is a pre-trained neural network model, which is fine-tuned on a custom dataset of plant images. The project also includes a user-friendly web application built with **Streamlit**, allowing anyone to upload an image and get an instant disease prediction.

---

## 🚀 Getting Started

To get the project running on your local machine, follow these steps.

### Prerequisites

You'll need to have **Python 3.8 or newer** and **pip** installed.

### Installation

1.  Clone the repository from GitHub:

    ```bash
    git clone [https://github.com/samalhitesh8-sudo/plant-disease-detection.git]
    ```

2.  Navigate into the project directory:

    ```bash
    cd PLANT
    ```

3.  Install all the required Python libraries using the `requirements.txt` file:

    ```bash
    pip install -r requirements.txt
    ```

---

## 💡 Usage

### Running the Streamlit Web Application

The easiest way to use the model is through the Streamlit web application.

1.  Start the Streamlit app from the project's root directory:

    ```bash
    streamlit run app.py
    ```

2.  Your default web browser will automatically open and navigate to the application. If not, open your browser and go to `http://localhost:8501`.

3.  Upload an image of a plant leaf. The application will use the pre-trained model to predict the disease and display the result on the screen.

### Retraining the Model

If you have a new dataset or want to improve the model, you can retrain it.

1.  Place your new image dataset inside the `data/dataset/` folder. Make sure the directory structure is suitable for the training script (e.g., organized into subfolders by class).

2.  Run the training script:

    ```bash
    python model/model_training.py
    ```

This script will train a new model and save it as `saved_model.h5` in the `model/` directory, overwriting the old one.

---

## 📁 Project Structure

* `app.py`: The main **Streamlit** application that serves the web interface.
* `model/`: Contains the trained deep learning model (`saved_model.h5`) and the script used for training (`model_training.py`).
* `data/`: Holds the dataset of plant images and their corresponding labels (`dataset_labels.csv`).
* `utils/`: Includes utility scripts for image preprocessing (`image_preprocessing.py`) and model inference (`inference.py`).
* `requirements.txt`: Lists all Python dependencies required for the project.

---

## 🤝 Contributing

We welcome contributions! If you'd like to help, please follow these steps:

1.  Fork the repository.
2.  Create your feature branch (`git checkout -b feature/your-feature-name`).
3.  Commit your changes (`git commit -m 'feat: Add a new feature'`).
4.  Push to the branch (`git push origin feature/your-feature-name`).
5.  Open a Pull Request.

---
