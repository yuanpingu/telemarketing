# Bank Telemarketing Prediction Dashboard
This supervised machine learning classification project predicts whether a customer will subscribe to a term deposit using machine learning models with historical data.

## Demo App
[![Open in Streamlit](https://img.shields.io/badge/Open%20in-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://telemarketing-v2oaebvxubshmrfh5u33xf.streamlit.app/)

## Dataset
This project was inspired by and developed using the Bank Marketing (2012) dataset from the UC Irvine Machine Learning Repository.

Dataset DOI: https://doi.org/10.24432/C5K306.

## Variables
- Age (Numerical) - Client's age
- Job (Categorical) - Type of job
- Marital (Categorical) - Marital status
- Education (Categorical) - Education level
- Default (Categorical) - Has credit in default?
- Balance (Categorical) - Average yearly balance
- Housing (Categorical) - Has housing loan?
- Loan (Categorical) - Has personal loan?
- Contact (Categorical) - Contact communication type
- Day of week (Categorical) - Last contact day of the month
- Month (Categorical) - Last contact month
- Duration (Numerical) - Last contact duration in seconds
- Campaign (Numerical) - Number of contacts performed during this campaign
- Pdays (Numerical) - Number of days since the last contact from porevious campaign
- Previous (Numerical) - Number of contacts performed before this campaign
- Poutcome (Categorical) - Outcome of the previous marketing campaign
- Y (Categorical) - Has the client subscribed to a term deposit?

## Features
- EDA dashboard
- Data cleaning
- Hyperparameter tuning
- 5-fold cross-validation
- Model evaluation and comparison

## Tools
- Python
- Streamlit
- pandas
- scikit-learn

## Models used
- Logistic Regression
- K Nearest Neighbor
- Random Forest
- Neural Network

## How to use the app
1. Install required packages using 'pip install -r requirements.txt'
2. Run the app using 'streamlit run streamlit_app.py'
3. Upload 1 csv file containing historical data seperated by ";"

## Further Reading
- [Python Documentation](https://docs.python.org/3/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [GridSearchCV Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)