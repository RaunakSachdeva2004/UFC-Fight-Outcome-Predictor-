# 🥊 UFC Fight Outcome Predictor using Neural Networks

![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white)

<p align="center">
  <img src="https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExc2hxMG5tZTd3Zjh2Z3p3emJlN2xzYXFtcndlZWhyMWF4aGk3aHhpNyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/AvvYYEmamjle51oMQ4/giphy.gif" alt="UFC Prediction Demo" width="600">
</p>

## 🎯 Project Overview

A full-stack web application that predicts the winner of UFC matches using a neural network. This educational demonstration showcases the complete ML pipeline from data preprocessing to deployment with a clean web interface.

## ✨ Features

- 🌐 **Web-Based Interface**: Intuitive UI built with HTML and Tailwind CSS
- ⚡ **Real-Time Predictions**: Instant fight outcome predictions based on fighter statistics
- 📊 **Confidence Scoring**: Model provides confidence percentages for predictions
- 🏗️ **Decoupled Architecture**: Clear separation between frontend, backend, and ML model
- 🤖 **Neural Network Powered**: Deep learning model trained on historical UFC data

## 🛠️ Technology Stack

- **Backend**: Python, Flask
- **Machine Learning**: TensorFlow, Keras, Scikit-learn
- **Data Processing**: Pandas, NumPy
- **Frontend**: HTML, Tailwind CSS, JavaScript
- **Model Persistence**: Joblib

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- UFC Master dataset from Kaggle

### Installation & Setup

1. **Clone the repository**:
```bash
git clone https://github.com/your-username/ufc-fight-predictor.git
cd ufc-fight-predictor
```

2. **Create and activate a virtual environment**:
```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install pandas tensorflow scikit-learn joblib flask flask-cors
```

4. **Download the dataset**:
   - Visit [UFC Fight Data on Kaggle](https://www.kaggle.com/datasets/rajeevw/ufcdata)
   - Download the `ufc-master.csv` file
   - Place it in the project root directory

5. **Train the neural network**:
```bash
python model_training.py
```
This generates:
- `ufc_prediction_model.h5` (trained Keras model)
- `preprocessor.joblib` (Scikit-learn preprocessor)

6. **Start the backend server**:
```bash
python app.py
```
Server runs at http://127.0.0.1:5000

7. **Open the frontend**:
   - Open `index.html` in your web browser
   - Enter fighter statistics or use default values
   - Click "Predict Winner" to see results!

## 🧠 Model Architecture

The neural network features:

- **Input Layer**: Varies based on feature count after preprocessing
- **Hidden Layers**:
  - Dense Layer: 128 neurons with ReLU activation
  - Dropout Layer: 30% dropout rate
  - Dense Layer: 64 neurons with ReLU activation
  - Dropout Layer: 30% dropout rate
- **Output Layer**: 1 neuron with Sigmoid activation (binary classification)

### Training Details

- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Test Accuracy**: 63-64% on unseen data
- **Task**: Binary classification (fighter red vs. fighter blue)

## 📈 Performance

The model achieves approximately 63-64% accuracy on the test set, providing a solid baseline for the complex task of fight prediction.

## 🔮 Future Improvements

- **Advanced Feature Engineering**: Incorporate striking accuracy, takedown defense, and submission attempts
- **Hyperparameter Tuning**: Experiment with learning rates, batch sizes, and architectures
- **Specialized Models**: Build separate models for different weight classes or fighter stances
- **Cloud Deployment**: Deploy to Heroku, AWS, or other cloud platforms
- **Real-time Data Integration**: Connect with live UFC statistics APIs
- **Historical Fighter Comparison**: Add fighter vs. fighter historical analysis

## 🤝 Contributing

We welcome contributions! Feel free to:
- Report bugs and issues
- Suggest new features and enhancements
- Submit pull requests
- Improve documentation

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">
⭐ Don't forget to star this repo if you found it useful! ⭐
</p>

<p align="center">
Made with ❤️ and 🥊 
</p>
