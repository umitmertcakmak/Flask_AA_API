from sklearn import linear_model
from sklearn import svm



def get_model(x):
    return {
        'LinearRegression' : linear_model.LinearRegression(),
        'Ridge' : linear_model.Ridge (alpha = .5),
        'RidgeCV' : linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0]),
        'Lasso' : linear_model.Lasso(alpha = 0.1),
        'LassoLars' : linear_model.LassoLars(alpha=.1),
        'BayesianRidge' : linear_model.BayesianRidge(),
        'SGDRegressor' : linear_model.SGDRegressor(),
        'SGDClassifier' : linear_model.SGDClassifier(),
        'SVM' : svm.SVC(),
        'SVR' : svm.SVR(),
    }[x]