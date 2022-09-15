
import joblib

from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
import sklearn

from azureml.core import Dataset
from azureml.core.run import Run
import numpy as np

from azureml.core import Model
from azureml.core.resource_configuration import ResourceConfiguration

X, y = load_diabetes(return_X_y=True)
model = Ridge().fit(X, y)

joblib.dump(model, 'sklearn_regression_model.pkl')

# Save datasets
np.savetxt('features.csv', X, delimiter=',')
np.savetxt('labels.csv', y, delimiter=',')

run = Run.get_context()
ws = run.experiment.workspace
datastore = ws.get_default_datastore()
datastore.upload_files(
    files=['./features.csv', './labels.csv'],
    target_path='sklearn_regression/',
    overwrite=True,
    )

input_dataset = Dataset.Tabular.from_delimited_files(path=[(datastore, 'sklearn_regression/features.csv')])
output_dataset = Dataset.Tabular.from_delimited_files(path=[(datastore, 'sklearn_regression/labels.csv')])

# Register model
model = Model.register(
    workspace=ws,
    model_name='my-sklearn-model',
    model_path='./sklearn_regression_model.pkl',
    model_framework=Model.Framework.SCIKITLEARN,
    model_framework_version=sklearn.__version__,
    sample_input_dataset=input_dataset,
    sample_output_dataset=output_dataset,
    resource_configuration=ResourceConfiguration(cpu=1, memory_in_gb=0.5),
    description='Ridge regression model to predictdisbetes progression.',
    tags={'area': 'diabetes', 'type': 'regression'},
)

print('Name:', model.name)
print('Version:', model.version)
