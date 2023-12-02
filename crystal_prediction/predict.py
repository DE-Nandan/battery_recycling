import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import tensorflow as tf
import keras
from keras.layers import Dense, BatchNormalization, Dropout, LSTM
from keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from keras import regularizers
from sklearn.compose import ColumnTransformer

import os
global preds

def predict_crystal_structure(crystal_instance,prediction_method):

    file_path = os.path.join(os.path.dirname(__file__), 'lithium-ion batteries.csv')

    data_train = pd.read_csv(file_path)
 
    

    data_train.drop(['Materials Id'], axis=1,inplace=True)

    data_train.dropna(inplace=True)

    lb_encoder_1 = LabelEncoder()
    lb_encoder_2 = LabelEncoder()
    lb_encoder_3 = LabelEncoder()
    data_train["Formula"] = lb_encoder_1.fit_transform(data_train["Formula"])
    data_train["Spacegroup"] = lb_encoder_2.fit_transform(data_train["Spacegroup"])
    data_train["Crystal System"] = lb_encoder_3.fit_transform(data_train["Crystal System"])


    X = data_train.drop('Crystal System',axis=1)
    y = data_train['Crystal System']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


    transformed_data = {
        'Formula': [crystal_instance.formula],
        'Spacegroup': [crystal_instance.spacegroup],
        'Formation Energy (eV)': [crystal_instance.formation_energy],
        'E Above Hull (eV)': [crystal_instance.e_above_hull],
        'Band Gap (eV)': [crystal_instance.band_gap],
        'Nsites': [crystal_instance.nsites],
        'Density (gm/cc)': [crystal_instance.density],
        'Volume': [crystal_instance.volume],
        'Has Bandstructure': [crystal_instance.has_bandstructure]
    }

    df = pd.DataFrame(transformed_data)

    df["Formula"] = lb_encoder_1.transform(df["Formula"])
    df["Spacegroup"] = lb_encoder_2.transform(df["Spacegroup"])


    prediction = ''

    if(prediction_method == 'xgb'):
        xgb_model = XGBClassifier()
        xgb_model.fit(X_train, y_train)
        prediction = xgb_model.predict(df)
    elif(prediction_method == 'random_forest'):
        rfc = RandomForestClassifier(min_samples_split = 15, min_samples_leaf = 20)
        rfc.fit(X_train, y_train)
        prediction = rfc.predict(df)
    elif(prediction_method == 'decision_tree'):
        decision_tree = DecisionTreeRegressor()
        decision_tree.fit(X_train, y_train)
        prediction = decision_tree.predict(df).astype(int)
    else:
        data = pd.read_csv(file_path)
        y = data['Crystal System']
        data.drop(['Materials Id', 'Crystal System'],
        axis=1,inplace=True)

        numerical_transformer = StandardScaler()
        label_transformer = OrdinalEncoder()

        n_cols = [c for c in data.columns if data[c].dtype in ['int64', 'float64', 'int32', 'float32']]
        obj_cols = [c for c in data.columns if data[c].dtype in ['bool', 'object']]
        
        ct = ColumnTransformer([('num', numerical_transformer, n_cols), ('non_num', label_transformer, obj_cols),])
        processed = ct.fit_transform(data)
        processed2 = ct.fit_transform(df)
        new_data = pd.DataFrame(columns=data.columns, data=processed)
        new_data2 = pd.DataFrame(columns=df.columns, data=processed2)
        X = new_data
        lb_encoder = LabelEncoder()
        y = lb_encoder.fit_transform(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
        model = Sequential()
        model.add(Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
        model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
        model.add(Dropout(0.2))
        model.add(Dense(128, activation='softsign', kernel_regularizer=regularizers.l2(0.01)))
        model.add(Dropout(0.2))
        model.add(Dense(3, activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        def train_model(n_runs, t_size=0.25):
            score = []
            for j in range(n_runs):
                X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), test_size=t_size, shuffle=True)
                y_encoded = to_categorical(y_train)
                
                # Fit the model
                history = model.fit(X_train, y_encoded, epochs=100, verbose=False, validation_split=0.2)
                
                # Evaluate the model on the test set
                preds = np.argmax(model.predict(X_test), axis=-1)
                score.append(accuracy_score(y_test, preds))
            
            return score
        
        scores = train_model(n_runs=5) 
        prediction = np.argmax(model.predict(np.array(new_data2)), axis=-1)



        




       
  
    predicted_crystal_system = lb_encoder_3.inverse_transform(prediction)[0]

    return predicted_crystal_system




