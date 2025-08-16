# Wellness Tourism Package Predictor

A machine learning application that predicts the likelihood of customers accepting wellness tourism packages based on their demographic and behavioral characteristics.

## 🎯 Project Overview

This project uses XGBoost classification to predict whether a customer will accept a wellness tourism package offer. The model considers various factors including customer demographics, travel preferences, financial status, and interaction history to make predictions.

## 🚀 Features

- **Interactive Web Application**: Streamlit-based UI for real-time predictions
- **Machine Learning Model**: XGBoost classifier with hyperparameter tuning
- **Model Deployment**: Containerized deployment with Docker
- **Cloud Integration**: Model hosted on Hugging Face Hub
- **Comprehensive Preprocessing**: Handles both numerical and categorical features

## 📊 Model Features

The model uses the following input features to make predictions:

### Demographic Features
- **Age**: Customer age (18-100 years)
- **Gender**: Male/Female
- **Marital Status**: Single/Married/Divorced
- **Occupation**: Salaried/Self Employed/Business Owner
- **Designation**: Executive/Managerial/Professional/Other
- **Monthly Income**: Customer's monthly income

### Travel Preferences
- **City Tier**: Tier 1/Tier 2/Tier 3
- **Number of Person Visiting**: Group size (1-10)
- **Preferred Property Star**: Hotel star preference (1-5)
- **Number of Trips**: Previous travel experience
- **Passport**: Yes/No
- **Own Car**: Yes/No
- **Number of Children Visiting**: Children in the group (0-10)

### Interaction Features
- **Type of Contact**: Company Invited/Self Inquiry
- **Product Pitched**: Basic/Deluxe/King/Standard/Super Deluxe
- **Duration of Pitch**: Pitch duration in minutes
- **Number of Followups**: Follow-up interactions (0-10)
- **Pitch Satisfaction Score**: Customer satisfaction (1-5)

## 🛠️ Technology Stack

- **Machine Learning**: XGBoost, Scikit-learn
- **Web Framework**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Model Deployment**: Docker, Hugging Face Hub
- **Development**: Python 3.9

## 📁 Project Structure

```
wellness_tourism_package_predictor/
├── tourism_project/
│   ├── deployment/
│   │   ├── app.py              # Streamlit web application
│   │   ├── Dockerfile          # Container configuration
│   │   └── requirements.txt    # Deployment dependencies
│   ├── hosting/
│   │   └── hosting.py          # Hosting utilities
│   ├── model_building/
│   │   ├── data_register.py    # Data registration utilities
│   │   ├── prep.py             # Data preprocessing
│   │   └── train.py            # Model training and tuning
│   └── requirements.txt        # Project dependencies
└── README.md
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Docker (optional, for containerized deployment)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd wellness_tourism_package_predictor
   ```

2. **Install dependencies**
   ```bash
   cd tourism_project
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application**
   ```bash
   cd deployment
   streamlit run app.py
   ```

4. **Access the application**
   Open your browser and navigate to `http://localhost:8501`

### Docker Deployment

1. **Build the Docker image**
   ```bash
   cd tourism_project/deployment
   docker build -t wellness-tourism-predictor .
   ```

2. **Run the container**
   ```bash
   docker run -p 8501:8501 wellness-tourism-predictor
   ```

## 🔧 Model Training

To retrain the model with new data:

1. **Prepare your data**
   - Ensure data is in the correct format with all required features
   - Split data into training and testing sets

2. **Run the training script**
   ```bash
   cd tourism_project/model_building
   python train.py
   ```

3. **Model artifacts**
   - The trained model will be saved as `best_tourism_project_model_v1.joblib`
   - Model is automatically uploaded to Hugging Face Hub

## 📈 Model Performance

The model uses the following techniques for optimal performance:

- **Hyperparameter Tuning**: GridSearchCV with 5-fold cross-validation
- **Class Imbalance Handling**: Scale_pos_weight parameter in XGBoost
- **Feature Engineering**: StandardScaler for numerical features, OneHotEncoder for categorical features
- **Evaluation Metric**: Recall score optimization for better positive class detection

## 🌐 Deployment

The application is designed for easy deployment:

- **Local Development**: Run with Streamlit
- **Containerized Deployment**: Docker support included
- **Cloud Deployment**: Compatible with cloud platforms (Heroku, AWS, GCP, Azure)
- **Model Hosting**: Pre-trained model available on Hugging Face Hub

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Contact

For questions or support, please open an issue in the repository.

## 🙏 Acknowledgments

- Dataset provided through Hugging Face Datasets
- Model hosted on Hugging Face Hub
- Built with Streamlit for interactive web interface