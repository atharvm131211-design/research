
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

data = pd.read_csv('/content/airfoil_self_noise.dat', sep='\s+', header=None)

data.columns = ['Frequency', 'Angle', 'Chord', 'Velocity', 'Thickness', 'Sound']

X = data[['Angle', 'Velocity', 'Chord', 'Thickness']]
y = data['Sound']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "Extra Trees": ExtraTreesRegressor(),
    "SVR": SVR(),
    "KNN": KNeighborsRegressor(),
    "Neural Network": MLPRegressor(max_iter=500)
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    error = mean_absolute_error(y_test, pred)
    print(name, ":", error)
    results.append([name, error])

df = pd.DataFrame(results, columns=["Model", "MAE"])
df.to_csv("/content/results.csv", index=False)

print("\nSaved results to results.csv")
