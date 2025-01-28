import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.metrics import confusion_matrix, precision_score, recall_score

def main():
    st.title('Machine Learning Web App')
    st.sidebar.title('Machine Learning Web App')

    #function decorator
    @st.cache_data
    def load_data():
        df = pd.read_csv(r'mushroom.csv', delimiter=";")
        label = LabelEncoder()

        #converting the values into numerical values
        for col in df.columns:
            df[col] = label.fit_transform(df[col])
        return df

    @st.cache_data # ensuring that data is read only once
    def split(df):
        y = df['type'].values.copy()
        X = df.drop(columns = ['type'])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
        
        return X_train, X_test, y_train, y_test
    

    def plot_metrics(metrics_list, model, X_test, y_test, y_pred):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(ax=ax)
            st.pyplot(fig)
            
        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            try:
                fig, ax = plt.subplots()
                RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"ROC Curve not available: {e}")
        
        if 'Precision-Recall Curve' in metrics_list:
            st.subheader('Precision-Recall Curve')
            try:
                fig, ax = plt.subplots()
                PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Precision-Recall Curve not available: {e}")

    df = load_data()
    X_train, X_test, y_train, y_test = split(df)

    st.sidebar.subheader('Choose a Classifier')
    classifier = st.sidebar.selectbox('Classifier', ('SVM', 'Logistic Regression', 'Random Forest', 'XGBoost', 'KNN'))

    if classifier == 'SVM':
        st.sidebar.subheader('Model hyperparameters')
        C = st.sidebar.number_input('C (Regularization parameter)', 0.01, 10.0, step = 0.01, key = 'C')
        Kernel = st.sidebar.radio('Kernel', ('rbf', 'linear'), key = 'Kernel')
        Gamma = st.sidebar.radio('Gamma (Kernel Coefficient)', ('scale', 'auto'), key = 'Gamma')

        metrics = st.sidebar.multiselect('What metrics to plot?', ('Confusion matrix', 'ROC Curve', 'Precision-Recall'))

        if st.sidebar.button('Classify', key = 'classify'):
            st.subheader('Support Vector Machine (SVM) Results')
            model = SVC(C = C, kernel = Kernel, gamma = Gamma)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.write('Accuracy:', round(model.score(X_test, y_test), 2))
            st.write('Precision:', round(precision_score(y_test, y_pred, average='macro'), 2))
            st.write('Recall:', round(recall_score(y_test, y_pred, average='macro'), 2))
            plot_metrics(metrics, model, X_test, y_test, y_pred)

    elif classifier == 'Logistic Regression':
        st.sidebar.subheader('Model Hyperparameters')
        C = st.sidebar.number_input('C (Regularization parameter)', 0.01, 10.0, step = 0.01, key = 'C_LR')
        max_iter = st.sidebar.number_input('Maximum number of iterations', 100, 500, step=50, key='max_iter')

        metrics = st.sidebar.multiselect('What metrics to plot?', ('Confusion Matrix', 'ROC Curve', 'Precision-Recall'))

        if st.sidebar.button('Classify', key = 'classify'):
            st.subheader('Logistic Regression Results')
            model = LogisticRegression(C = C, max_iter = max_iter)
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)
            y_pred = model.predict(X_test)
            st.write('Accuracy:', round(model.score(X_test, y_test), 2))
            st.write('Precision:', round(precision_score(y_test, y_pred, average='macro'), 2))
            st.write('Recall:', round(recall_score(y_test, y_pred, average='macro'), 2))
            plot_metrics(metrics, model, X_test, y_test, y_pred)

    elif classifier == 'Random Forest':
        st.sidebar.subheader('Model Hyperparameters')
        n_estimators = st.sidebar.number_input('The number of trees in the forest', 100, 5000, step = 10, key  = 'n_estimators')
        max_depth  = st.sidebar.number_input('The maximum depth of the tree', 1, 20, step = 1, key = 'max_depth')
        bootstrap = st.sidebar.radio('Bootstrap samples when building trees ', (True, False), key = 'bootstrap' )
        metrics = st.sidebar.multiselect('What metrics to plot?', ('Confusion Matrix', 'ROC Curve', 'Precision-Recall'))

        if st.sidebar.button('Classify', key = 'classify'):
            st.subheader('Random Forest Results')
            model = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth, bootstrap = bootstrap, n_jobs = -1)
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)
            y_pred = model.predict(X_test)
            st.write('Accuracy:', round(model.score(X_test, y_test), 2))
            st.write('Precision:', round(precision_score(y_test, y_pred, average='macro'), 2))
            st.write('Recall:', round(recall_score(y_test, y_pred, average='macro'), 2))
            plot_metrics(metrics, model, X_test, y_test, y_pred)
    
    elif classifier == 'XGBoost':
        st.sidebar.subheader('Model Hyperparameters')
        learning_rate = st.sidebar.slider('Learning Rate', 0.01, 0.5, step=0.01, key='learning_rate')
        n_estimators = st.sidebar.slider('Number of Trees', 50, 500, step=50, key='xgb_n_estimators')

        metrics = st.sidebar.multiselect('What metrics to plot?', ('Confusion Matrix', 'ROC Curve', 'Precision-Recall'))

        if st.sidebar.button('Classify', key='classify'):
            st.subheader('XGBoost Results')
            model = XGBClassifier(learning_rate=learning_rate, n_estimators=n_estimators, use_label_encoder=False, eval_metric='logloss')
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.write('Accuracy:', round(model.score(X_test, y_test), 2))
            st.write('Precision:', round(precision_score(y_test, y_pred, average='macro'), 2))
            st.write('Recall:', round(recall_score(y_test, y_pred, average='macro'), 2))

            plot_metrics(metrics)
    
    elif classifier == 'KNN':
        st.sidebar.subheader('Model Hyperparameters')
        n_neighbors = st.sidebar.slider('Number of Neighbors', 1, 15, step=1, key='n_neighbors')
        
        metrics = st.sidebar.multiselect('What metrics to plot?', ('Confusion Matrix', 'ROC Curve', 'Precision-Recall'))

        if st.sidebar.button('Classify', key='classify'):
            st.subheader('K-Nearest Neighbors (KNN) Results')
            model = KNeighborsClassifier(n_neighbors=n_neighbors)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            st.write('Accuracy:', round(model.score(X_test, y_test), 2))
            st.write('Precision:', round(precision_score(y_test, y_pred, average='macro'), 2))
            st.write('Recall:', round(recall_score(y_test, y_pred, average='macro'), 2))
            
            plot_metrics(metrics, model, X_test, y_test, y_pred)

    if st.sidebar.checkbox('Show raw data ', False):
        st.subheader('Mushroon Data Set (Classification)')
        st.write(df)

if __name__ == "__main__":
    main()