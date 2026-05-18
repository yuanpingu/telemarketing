import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.title("Bank Marketing Data Analytics Dashboard", text_alignment="center")

st.write("This app aims to predict a customer subscribing to a long term deposit using historical data")

uploaded_file = st.file_uploader("Upload the CSV file of customer information (data must be seperated by ' ; ')", type=["csv"])

if uploaded_file is None:
    st.info("Please upload a csv file to begin.")
    st.stop()

st.space(40)



#Introduction
st.header("Introduction to the dataset", divider=True,text_alignment="center")
df = pd.read_csv(uploaded_file, sep=";")

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Dataset Shape")
st.write(f"Rows: {df.shape[0]}")
st.write(f"Columns: {df.shape[1]}")

st.subheader("Column Names")
st.write(df.columns.tolist())

st.subheader("Missing Values")
st.write(df.isnull().sum())
st.write("Rows with missing values will be :red[removed!]")
df.dropna()

st.subheader("Duplicated Rows")
st.write(df.duplicated().sum())
st.write("Duplicated rows will be :red[removed!]")
df.drop_duplicates()

st.space(40)



#EDA on variables
st.header("Exploratory Data Analysis on variables", divider=True,text_alignment="center")
st.subheader("Target variable distribution")
target_col = "y"
fig, ax = plt.subplots()
df[target_col].value_counts().plot(kind="bar", ax=ax)
ax.set_xlabel("Subscribed to Term Deposit")
ax.set_ylabel("Count")
st.pyplot(fig)

target_counts = df[target_col].value_counts()
target_percent = df[target_col].value_counts(normalize=True) * 100
target_summary = pd.DataFrame({"Count": target_counts, "Percentage (%)": target_percent.round(2)})
st.dataframe(target_summary)



#Check for imbalance in dataset
minority_percent = target_percent.round(2).min()
st.write(f"Minority class percentage: {minority_percent}%")

if minority_percent < 30:
    st.warning("The dataset is :red[imbalanced], hence :red[F1-score] will be the main evaluation metric.")
    main_metric = "F1-score"
    cv_scoring = "f1"
else:
    st.success("The dataset is relatively balanced, hence :red[accuracy] will be used as the main evaluation metric.")
    main_metric = "Accuracy"
    cv_scoring = "accuracy"



#Visualizing each variable
st.subheader("Variable visualisation")

numerical_col = df.select_dtypes(include=["int64", "float64"]).columns.to_list()
categorical_col = df.select_dtypes(include=["object", "string"]).columns.to_list()

plot_type = st.selectbox("Choose variable type", ["Categorical Bar Chart", "Numerical Boxplot"])

if plot_type == "Categorical Bar Chart":
    selected_cat = st.selectbox("Choose a categorical variable", categorical_col, key = "categorical_bar_selection")
    subscription_counts = df.groupby([selected_cat, "y"]).size().unstack()
    st.bar_chart(subscription_counts)

elif plot_type == "Numerical Boxplot":
    selected_num = st.selectbox("Choose a numerical variable", numerical_col)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.boxplot(df[selected_num], vert=False)
    ax.set_title(f"Boxplot of {selected_num}")
    ax.set_xlabel(selected_num)
    st.pyplot(fig)



#Data cleaning and splitting
df_model = df.copy()

y_first_value = str(df_model[target_col].iloc[0]).lower()
if y_first_value in ["yes", "no"]: #mapping target variable to 0 and 1 
    df_model[target_col] = (df_model[target_col].astype(str).str.lower().map({"no": 0, "yes": 1}))

df_model = pd.get_dummies(df_model, drop_first=True) #one-hot encoding categorical variables

X = df_model.drop(target_col, axis=1)
target = df_model[target_col]

X_train, X_test, target_train, target_test = train_test_split(
    X,
    target,
    test_size=0.2,
    random_state=42,
    stratify=target
)

scalar = StandardScaler()#standardizing numerical variables
X_train[numerical_col] = scalar.fit_transform(X_train[numerical_col])
X_test[numerical_col] = scalar.fit_transform(X_test[numerical_col])

st.space(40)




#Logistic Regression Model 
st.header("Logistic Regression Model", divider=True, text_alignment="center")
log_model = LogisticRegression(max_iter=300)

param_log = {"C": [0.01, 0.1, 1, 10]} #cross validation to find the best parameters for C
log_cv = GridSearchCV(
    estimator=log_model,
    param_grid=param_log, 
    scoring=cv_scoring, 
    cv=5, 
    n_jobs=-1)
log_cv.fit(X_train, target_train)

log_cv_results = pd.DataFrame(log_cv.cv_results_)

log_cv_scores = log_cv_results[["param_C", "mean_test_score"]]

st.subheader("C Value vs Cross-Validation Score") #graph to visualize the trend
st.dataframe(log_cv_scores)

fig, ax = plt.subplots()

ax.plot(log_cv_scores["param_C"], log_cv_scores["mean_test_score"], marker="o")
ax.set_xlabel("C value")
ax.set_ylabel("Mean CV Score")
ax.set_title("Logistic Regression: C vs CV Score")
ax.set_xscale("log")

st.pyplot(fig)


#Choosing the best LR model to use on test set
best_log = log_cv.best_estimator_
y_pred_log = best_log.predict(X_test)

log_accuracy = accuracy_score(target_test, y_pred_log)
log_precision = precision_score(target_test, y_pred_log)
log_recall = recall_score(target_test, y_pred_log)
log_f1 = f1_score(target_test, y_pred_log)

st.subheader("Best parameters found using 5-fold cross-validation:")
st.markdown(f""" 
        * C = `{log_cv.best_params_["C"]}` 
        * Best CV score = `{(round(log_cv.best_score_, 4))}`
            """)

st.subheader("Evaluation:")
left_col, right_col = st.columns(2)

with left_col:
    st.write(f"Accuracy: {log_accuracy: .4f} = :green[{log_accuracy: .2%}]")
    st.write(f"Precision: {log_precision: .4f} = :green[{log_precision: .2%}]")

with right_col: 
    st.write(f"Recall: {log_recall: .4f} = :green[{log_recall: .2%}]")
    st.write(f"F1-score: {log_f1: .4f} = :green[{log_f1: .2%}]")

if main_metric == "Accuracy":
    selected_score = round(log_accuracy, 4)
else:
    selected_score = round(log_f1, 4)

st.success(f"Logistic Regression {main_metric}: {selected_score} = **{selected_score * 100:.2f}%**")

st.space(40)



#kNN Model
st.header("k Nearest Neighbors Model", divider=True, text_alignment="center")
knn_model = KNeighborsClassifier()


param_knn = {"n_neighbors": [3, 5, 7, 9, 11]} #cross validation to find the best parameters for k
knn_cv = GridSearchCV(
    estimator=knn_model, 
    param_grid=param_knn, 
    scoring=cv_scoring, 
    cv=5, 
    n_jobs=-1)
knn_cv.fit(X_train, target_train)

knn_cv_results = pd.DataFrame(knn_cv.cv_results_)

knn_cv_scores = knn_cv_results[["param_n_neighbors", "mean_test_score"]]

st.subheader("k Value vs Cross-Validation Score") #graph to visualize the trend
st.dataframe(knn_cv_scores)

fig, ax = plt.subplots()

ax.plot(knn_cv_scores["param_n_neighbors"], knn_cv_scores["mean_test_score"], marker="o")
ax.set_xlabel("k value")
ax.set_ylabel("Mean CV Score")
ax.set_title("kNN: k vs CV Score")

st.pyplot(fig)

#Choosing the best knn model to use on test set
best_knn = knn_cv.best_estimator_
y_pred_knn = best_knn.predict(X_test)

knn_accuracy = accuracy_score(target_test, y_pred_knn)
knn_precision = precision_score(target_test, y_pred_knn)
knn_recall = recall_score(target_test, y_pred_knn)
knn_f1 = f1_score(target_test, y_pred_knn)

st.subheader("Best parameters found using 5-fold cross-validation:")
st.markdown(f""" 
        * k = `{knn_cv.best_params_["n_neighbors"]}` 
        * Best CV score = `{(round(knn_cv.best_score_, 4))}`
            """)
st.info("Too small of a k value can lead to overfitting, hence the selected k value should be judged using test-set performance")

st.subheader("Evaluation:")
left_col, right_col = st.columns(2)

with left_col:
    st.write(f"Accuracy: {knn_accuracy: .4f} = :green[{knn_accuracy: .2%}]")
    st.write(f"Precision: {knn_precision: .4f} = :green[{knn_precision: .2%}]")

with right_col: 
    st.write(f"Recall: {knn_recall: .4f} = :green[{knn_recall: .2%}]")
    st.write(f"F1-score: {knn_f1: .4f} = :green[{knn_f1: .2%}]")


if main_metric == "Accuracy":
    selected_score = round(knn_accuracy, 4)
else:
    selected_score = round(knn_f1, 4)

st.success(f"k Nearest Neighbors {main_metric}: {selected_score} = **{selected_score * 100:.2f}%**")

st.space(40)




#Random Forest Model
st.header("Random Forest Model", divider=True, text_alignment="center")
rf_model = RandomForestClassifier(random_state=42, class_weight="balanced")

param_rf = {"n_estimators": [100, 200, 300, 400, 500]} #cross validation to find the best parameters for estimators
rf_cv = GridSearchCV(
    estimator=rf_model, 
    param_grid=param_rf, 
    scoring=cv_scoring, 
    cv=5, 
    n_jobs=-1)
rf_cv.fit(X_train, target_train)

rf_cv_results = pd.DataFrame(rf_cv.cv_results_)

rf_cv_scores = rf_cv_results[["param_n_estimators", "mean_test_score"]]

st.subheader("Number of estimators vs Cross-Validation Score") #graph to visualize the trend
st.dataframe(rf_cv_scores)

fig, ax = plt.subplots()

ax.plot(rf_cv_scores["param_n_estimators"], rf_cv_scores["mean_test_score"], marker="o")
ax.set_xlabel("Number of estimators")
ax.set_ylabel("Mean CV Score")
ax.set_title("Random Forest: Number of estimators vs CV Score")

st.pyplot(fig)

#Choosing the best rf model to use on test set
best_rf = rf_cv.best_estimator_
y_pred_rf = best_rf.predict(X_test)

rf_accuracy = accuracy_score(target_test, y_pred_rf)
rf_precision = precision_score(target_test, y_pred_rf)
rf_recall = recall_score(target_test, y_pred_rf)
rf_f1 = f1_score(target_test, y_pred_rf)

st.subheader("Best parameters found using 5-fold cross-validation:")
st.markdown(f""" 
        * Number of estimators = `{rf_cv.best_params_["n_estimators"]}` 
        * Best CV score = `{(round(rf_cv.best_score_, 4))}`
            """)

st.subheader("Evaluation:")
left_col, right_col = st.columns(2)

with left_col:
    st.write(f"Accuracy: {rf_accuracy: .4f} = :green[{rf_accuracy: .2%}]")
    st.write(f"Precision: {rf_precision: .4f} = :green[{rf_precision: .2%}]")

with right_col: 
    st.write(f"Recall: {rf_recall: .4f} = :green[{rf_recall: .2%}]")
    st.write(f"F1-score: {rf_f1: .4f} = :green[{rf_f1: .2%}]")

if main_metric == "Accuracy":
    selected_score = round(rf_accuracy, 4)
else:
    selected_score = round(rf_f1, 4)

st.success(f"Random Forest Model {main_metric}: {selected_score} = **{selected_score * 100:.2f}%**")

st.space(40)




#Neural Network Model

st.header("Neural Network Model", divider=True, text_alignment="center")
nn_model = MLPClassifier(activation="relu", random_state=42, max_iter=1000, early_stopping=True)


param_nn = {"hidden_layer_sizes": [(64, 32), (64, 32, 16), (128, 64, 32), (128, 64, 32, 16)]} #cross validation to find the best parameters for hidden layers
nn_cv = GridSearchCV(
    estimator=nn_model, 
    param_grid=param_nn, 
    scoring=cv_scoring, 
    cv=5, 
    n_jobs=-1)
nn_cv.fit(X_train, target_train)

nn_cv_results = pd.DataFrame(nn_cv.cv_results_)

nn_cv_scores = nn_cv_results[["param_hidden_layer_sizes", "mean_test_score"]]

st.subheader("Sizes of hidden layers vs Cross-Validation Score")
st.dataframe(nn_cv_scores)

fig, ax = plt.subplots()

ax.bar(
    nn_cv_scores["param_hidden_layer_sizes"].astype(str),
    nn_cv_scores["mean_test_score"]
)

ax.set_xlabel("Hidden layer sizes")
ax.set_ylabel("Mean CV Score")
ax.set_title("Neural Network: Size of hidden layers vs CV Score")
ax.tick_params(axis="x", rotation=45)

st.pyplot(fig)

#Choosing the best nn model to use on test set
best_nn = nn_cv.best_estimator_
y_pred_nn = best_nn.predict(X_test)

nn_accuracy = accuracy_score(target_test, y_pred_nn)
nn_precision = precision_score(target_test, y_pred_nn)
nn_recall = recall_score(target_test, y_pred_nn)
nn_f1 = f1_score(target_test, y_pred_nn)

st.subheader("Best parameters found using 5-fold cross-validation:")
st.markdown(f""" 
        * Size of hidden layers = `{nn_cv.best_params_["hidden_layer_sizes"]}` 
        * Best CV score = `{(round(nn_cv.best_score_, 4))}`
            """)

st.subheader("Evaluation:")
left_col, right_col = st.columns(2)

with left_col:
    st.write(f"Accuracy: {nn_accuracy: .4f} = :green[{nn_accuracy: .2%}]")
    st.write(f"Precision: {nn_precision: .4f} = :green[{nn_precision: .2%}]")

with right_col: 
    st.write(f"Recall: {nn_recall: .4f} = :green[{nn_recall: .2%}]")
    st.write(f"F1-score: {nn_f1: .4f} = :green[{nn_f1: .2%}]")

if main_metric == "Accuracy":
    selected_score = round(nn_accuracy, 4)
else:
    selected_score = round(nn_f1, 4)

st.success(f"Neural Network Model {main_metric}: {selected_score} = **{selected_score * 100:.2f}%**")

st.space(40)




#Summary
st.header("Final results", divider=True, text_alignment="center")

if main_metric == "Accuracy":
    metric = "Accuracy"
    res_df = pd.DataFrame({'Model': ['Logistic Regression', 'kNN', 'Random Forest', 'Neural Network'], 
                       'Accuracy': [log_accuracy, knn_accuracy, rf_accuracy, nn_accuracy]})
else:
    metric = "F1"
    res_df = pd.DataFrame({'Model': ['Logistic Regression', 'kNN', 'Random Forest', 'Neural Network'], 
                       'F1': [log_f1, knn_f1, rf_f1, nn_f1]})
st.table(res_df)

best_row = res_df.loc[res_df[metric].idxmax()]
best_model_name = best_row["Model"]

best_params = {
    "Logistic Regression": log_cv.best_params_,
    "kNN": knn_cv.best_params_,
    "Random Forest": rf_cv.best_params_,
    "Neural Network": nn_cv.best_params_
}

params = best_params[best_model_name]

params_text = ", ".join(
    [f"{key} = {value}" for key, value in params.items()]
)


st.success(f"The best model based on **{main_metric}** is **{best_row['Model']}** "
           f"with {params_text}, getting a score of **{best_row[metric]:.4f}**.")
