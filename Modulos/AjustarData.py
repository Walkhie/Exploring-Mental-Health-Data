import pandas as pd
import numpy as np

# Funcion que imputara a lo que llamamos nulos con sentido
def ajustedData(df : pd.DataFrame) -> pd.DataFrame:

    ## Vamos inputar en Academic Pressure, Study Satisfaction y CGPA -1 en donde "Working Professional or Student" sea igual a
    ## Working Professional, de esta manera se interpretara que no aplica

    # Filtramos por los datos nulos en las columnas
    null_mask = df[['Academic Pressure', 'Study Satisfaction', 'CGPA']].isnull().any(axis=1)

    # Juntamos el condicion de null_mask con la de Student
    student_null_mask = (df['Working Professional or Student'] == 'Working Professional') & null_mask

    # Imputamos donde se cumpla ambas condiciones
    df.loc[student_null_mask, ['Academic Pressure', 'Study Satisfaction', 'CGPA']] = -1

    ## Vamos inputar en Job Satisfaction y Work Pressure -1 en donde "Working Professional or Student" sea igual a
    ## Student, de esta manera se interpretara que no aplica

    # Filtramos por los datos nulos en las columnas
    null_mask2 = df[['Job Satisfaction', 'Work Pressure', 'Profession']].isnull().any(axis=1)

    # Juntamos el condicion de null_mask con la de Student
    student_null_mask2 = (df['Working Professional or Student'] == 'Student') & null_mask2

    # Imputamos donde se cumpla ambas condiciones
    df.loc[student_null_mask2, ['Job Satisfaction', 'Work Pressure']] = -1

    ## Para Profession vamos imputar con "Student"  en donde "Working Professional or Student" sea igual a
    ## Student

    # Imputamos donde se cumpla ambas condiciones
    df.loc[student_null_mask2, ['Profession']] = 'Student'

    return df

# Función para clasificar la duración del sueño en 3 categorías
def categorizar_sueño(duracion):
  if isinstance(duracion, str):
    duracion = duracion.lower()
    if 'no' in duracion or duracion in ['pune', 'indore', 'sleep_duration', 'unhealthy', 'work_study_hours', 'moderate']:
        return np.nan  # Asignar nulo  los valores ambiguos
    elif any(h in duracion for h in ['less than 5 hours', '3-4', '2-3', '4-5', '1-6', '1-2', '1-3']):
        return 'Less than 5 hours'
    elif any(h in duracion for h in ['5-6', '6-7', '7-8', '6-8', '8 hours', '5-8']):
        return '5-8 hours'
    elif any(h in duracion for h in ['more than 8', '9-11', '10-11', '8-9']):
        return 'More than 8 hours'
    else:
        return np.nan  # Para valores numéricos que no caen en las categorías de sueño

  # Asigna nulo  a valores no reconocidos
  return np.nan


# Transformador para aplicar la función sobre la columna "Sleep Duration"
def transformar_sueño(df):
    df['Sleep Duration'] = df['Sleep Duration'].apply(categorizar_sueño)
    return df

def categorizar_dieta(habito):
    # Verifica si 'habito' es una cadena antes de proceder
    if isinstance(habito, str):
        habito = habito.lower()
        if 'unhealthy' in habito:
            return 'Unhealthy'
        elif 'healthy' in habito:
            if 'less' in habito or 'no' in habito or 'less than' in habito:
                return 'Moderate'
            elif 'more' in habito:
                return 'Healthy'
            else:
                return 'Healthy'
        elif 'moderate' in habito:
            return 'Moderate'
    # Asigna nulo  a valores no reconocidos
    return np.nan

# Transformador para aplicar la función de categorización
def transformar_dieta(df):
    df['Dietary Habits'] = df['Dietary Habits'].apply(categorizar_dieta)
    return df

def reduccion_categorias(df, threshold = 5000):
  # Agrupar categorías poco frecuentes en "Others"
  # Obtener categorías que superan el umbral
    frequent_degrees = df['Degree'].value_counts()[lambda x: x > threshold].index
    frequent_professions = df['Profession'].value_counts()[lambda x: x > threshold].index

    # Usar .map() para asignar 'Other' a las categorías menos frecuentes
    df['Degree'] = df['Degree'].map(lambda x: x if x in frequent_degrees else 'Other')
    df['Profession'] = df['Profession'].map(lambda x: x if x in frequent_professions else 'Other')

    return df

from sklearn.base import BaseEstimator, TransformerMixin

# Paso 1: Imputación
class Imputacion(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return ajustedData(X)  # Función que realiza la imputación de datos nulos

# Paso 2: Transformación de Sueño
class TransformarSueño(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return transformar_sueño(X)  # Función que reduce las categorías de "Sleep Duration"

# Paso 3: Transformación de Dieta
class TransformarDieta(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return transformar_dieta(X)  # Función que reduce las categorías de "Dietary Habits"

# Paso 4: Reducir las categorias de Degree y Profession

class ReduccionCategorias(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return reduccion_categorias(X)  # Función que reduce las categorías de "Degree" y "Profession"
    

def eliminar_variables_irrelevantes(df):
    return df.drop(['id','Name','Gender','Family History of Mental Illness','Sleep Duration','CGPA','Study Satisfaction','Job Satisfaction','Work Pressure'], axis=1)

def eliminar_outliers(X):
    #Se asigna valor nulo a los atípicos
    X.loc[~X["Academic Pressure"].isin([1, 2, 3, 4, 5, -1]), "Academic Pressure"] = np.nan
    X.loc[~X["Financial Stress"].isin([1, 2, 3, 4, 5]), "Financial Stress"] = np.nan
    X.loc[~X['Dietary Habits'].isin(['Healthy', 'Unhealthy', 'Moderate']), 'Dietary Habits'] = np.nan
    X.loc[~X['Degree'].isin(['Other', 'Class 12', 'B.Ed','B.Arch','B.Com','B.Pharm', 'BCA', 'M.Ed', 'MCA', 'BBA', 'BSc']), 'Degree'] = np.nan
    X.loc[~X['Profession'].isin(['Other', 'Teacher', 'Student','Content Writer']), 'Profession'] = np.nan
    X.loc[~X['Have you ever had suicidal thoughts ?'].isin(['Yes','No']), 'Have you ever had suicidal thoughts ?'] = np.nan
    return X
