from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomCityClassifier

numeric_features = ['salary', 'zone_count', 'staff_count']
categorical_features = ['rank', 'district']

categorical_feature_mask = df.dtypes==object
categorical_features = df.columns[categorical_feature_mask].tolist()

numeric_feature_mask = df.dtypes!=object
numeric_features = df.columns[numeric_feature_mask].tolist()

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore')), ])

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()), ])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features) ])

clf = Pipeline([
     ('preprocessor', preprocessor),
     ('clf', RandomCityClassifier()) ])
 
# Source https://adhikary.net/2019/03/23/categorical-and-numeric-data-in-scikit-learn-pipelines

Random Forest Importance
https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html

#Pipeline with multi cat and numeric cols
https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html

Imputation
https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html

Random Forest Examples
https://nonusingh.github.io/RandomForest
https://towardsdatascience.com/random-forest-ca80e56224c1

Pipeline examples
https://machinelearningmastery.com/modeling-pipeline-optimization-with-scikit-learn/
https://www.askpython.com/python/examples/pipelining-in-python
https://www.youtube.com/watch?v=xIqX1dqcNbY
https://www.youtube.com/watch?v=jzKSAeJpC6s&ab_channel=Dr.DataScience

Confusion Matrix eg
https://confusionmatrixonline.com/
