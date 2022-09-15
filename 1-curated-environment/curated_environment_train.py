
import joblib
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
import sklearn

X, y = load_diabetes(return_X_y=True)
print('Model Training...')
model = Ridge().fit(X, y)

joblib.dump(model, 'sklearn_ridge_model.pkl')

print('Model Trained!')
