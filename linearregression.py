import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

x = np.array([1996, 1998, 2000, 2002, 2004, 2006])
y = np.array([44, 69, 109, 141, 182, 233])

model = LinearRegression()
model.fit(x.reshape(-1, 1), y)

def predict_churn(data):
    data = np.array(data).reshape(-1, 1)
    return model.predict(data)

def plot_churn_predictions():
    years = np.array([1996, 1998, 2000, 2002, 2004, 2006, 2008]).reshape(-1, 1)
    predictions = model.predict(years)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=years.flatten(), y=predictions, color='blue', label='Predicted Churn')
    plt.plot(years.flatten(), predictions, color='red', linestyle='-', label='Trend Line')
    plt.title('Churn Prediction Over Years')
    plt.xlabel('Year')
    plt.ylabel('Predicted Churn')
    plt.legend()
    plt.show()
plot_churn_predictions()
print(predict_churn([2001, 2005]))