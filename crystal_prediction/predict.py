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
import os

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
       

    predicted_crystal_system = lb_encoder_3.inverse_transform(prediction)[0]

    return predicted_crystal_system




