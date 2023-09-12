import pandas as pd

# Carica il dataset sintetico generato da CTGAN
synthetic_data = pd.read_csv('synthetic_data.csv')

# Carica il dataset reale
real_data = pd.read_csv('KDDTrain+.csv')

# Carica il dataset per il test
test_data = pd.read_csv('KDDTest+.csv')


from sdv.metadata import SingleTableMetadata

# Carica i metadati 
metadata = SingleTableMetadata.load_from_json(
    filepath='metadati.json')


from sdv.evaluation.single_table import evaluate_quality

# Genrazione del Quality Report
report = evaluate_quality(
    real_data=real_data,
    synthetic_data=synthetic_data,
    metadata=metadata
)

# Generazione dell'istogramma relativo alla proprietà 'column shapes'
fig = report.get_visualization(property_name='Column Shapes')
fig.show()

# Generazione delle matrici di correlazione
fig = report.get_visualization(property_name='Column Pair Trends')
fig.show()



# METRICHE MLEfficacy 
from sdmetrics.single_table import BinaryAdaBoostClassifier, BinaryDecisionTreeClassifier, BinaryLogisticRegression, BinaryMLPClassifier    

# Decision Tree
BinaryDecisionTreeClassifier.compute(
    test_data=synthetic_data,
    train_data=test_data,
    target="'class'",
    metadata=metadata
)

# AdaBoost
BinaryAdaBoostClassifier.compute(
    test_data=synthetic_data,
    train_data=test_data,
    target="'class'",
    metadata=metadata
)

#Logistic Regression Classifier
BinaryLogisticRegression.compute(
    test_data=synthetic_data,
    train_data=test_data,
    target="'class'",
    metadata=metadata
)

#MLP Classifier
BinaryMLPClassifier.compute(
    test_data=synthetic_data,
    train_data=test_data,
    target="'class'",
    metadata=metadata
)


