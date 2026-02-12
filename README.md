# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import required Python libraries.
2. Read the salary dataset from the CSV file.
3. Select input feature(s) and target salary column.
4. Split the dataset into training and testing sets.
5. Train the Decision Tree Regressor model.
6. Predict salary values for test data.
7. Evaluate the model using MSE and R² score.
8. Visualize the decision tree.
9. Predict salary for a new employee.

## Program:

Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

Developed by: Mahalakshmi B

RegisterNumber:  212224040182

```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv("Salary.csv")
print("Employee Salary Data:\n", df, "\n")

X = df[['Level']]
y = df['Salary']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = DecisionTreeRegressor(criterion='squared_error', max_depth=3, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

plt.figure(figsize=(12, 6))
plot_tree(
    model,
    feature_names=X.columns,
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Decision Tree Regressor for Salary Prediction")
plt.show()
new_emp = [[6]]
predicted_salary = model.predict(new_emp)
print("\nPredicted Salary for New Employee:", round(predicted_salary[0], 2))
```
## Output:

<img width="758" height="337" alt="image" src="https://github.com/user-attachments/assets/b2f6cb56-f1ea-4e44-868b-b20fd739c5cf" />
<img width="1192" height="675" alt="image" src="https://github.com/user-attachments/assets/65d1f3f1-38c7-4576-936e-503c35200987" />


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
