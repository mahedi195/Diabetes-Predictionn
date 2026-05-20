from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv(r'C:\Users\user\Desktop\DiabtesPrediction\DIABETES\diabetes.csv')
X = data.drop('Outcome', axis=1)
Y = data['Outcome']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000) # Increased max_iter to prevent convergence warnings
model.fit(X_train, Y_train)


def home(request):
    return render(request, 'home.html')


def predict(request):
    return render(request, 'predict.html')


def result(request):
    raw_vals = [request.GET.get(f'n{i}', '0') for i in range(1, 9)]
    
    vals = []
    for val in raw_vals:
        try:
            vals.append(float(val) if val.strip() != "" else 0.0)
        except ValueError:
            vals.append(0.0)

    pred = model.predict([vals])

    if pred[0] == 1:
        result_text = "Diabetic"
    else:
        result_text = "Non-Diabetic"

    return render(request, "predict.html", {"result2": result_text})
