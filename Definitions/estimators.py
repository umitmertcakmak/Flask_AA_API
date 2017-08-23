estimators = [
    {
        'id': 1,
        'learning_type': 'Supervised Learning',
        'model_category': 'Generalized Linear Models',
        'model': 'LinearRegression',
        'description': 'LinearRegression fits a linear model with coefficients w = (w_1, ..., w_p) to minimize the residual sum of squares between the observed responses in the dataset, and the responses predicted by the linear approximation.',
        'tags': ['supervised', 'linear', 'regression'],
        'connection': {
          'istarget': True,
          'issource': False,
          'maxtargets': 3,
          'maxsources': 0
        }
    },
    {
        'id': 2,
        'learning_type': 'Supervised Learning',
        'model_category': 'Generalized Linear Models',
        'model': 'Ridge',
        'description': 'Ridge regression addresses some of the problems of Ordinary Least Squares by imposing a penalty on the size of coefficients.',
        'tags': ['supervised', 'ridge', 'regression'],
        'connection': {
          'istarget': True,
          'issource': False,
          'maxtargets': 3,
          'maxsources': 0
        }
    },
    {
        'id': 3,
        'learning_type': 'Supervised Learning',
        'model_category': 'Generalized Linear Models',
        'model': 'RidgeCV',
        'description': 'RidgeCV implements ridge regression with built-in cross-validation of the alpha parameter. The object works in the same way as GridSearchCV except that it defaults to Generalized Cross-Validation (GCV), an efficient form of leave-one-out cross-validation',
        'tags': ['supervised', 'ridge', 'regression', 'cross validation'],
        'connection': {
          'istarget': True,
          'issource': False,
          'maxtargets': 3,
          'maxsources': 0
        }
    },
    {
        'id': 4,
        'learning_type': 'Supervised Learning',
        'model_category': 'Generalized Linear Models',
        'model': 'Lasso',
        'description': 'The Lasso is a linear model that estimates sparse coefficients. It is useful in some contexts due to its tendency to prefer solutions with fewer parameter values, effectively reducing the number of variables upon which the given solution is dependent. For this reason, the Lasso and its variants are fundamental to the field of compressed sensing. Under certain conditions, it can recover the exact set of non-zero weights',
        'tags': ['supervised', 'Lasso', 'regression'],
        'connection': {
          'istarget': True,
          'issource': False,
          'maxtargets': 3,
          'maxsources': 0
        }

    },
    {
        'id': 5,
        'learning_type': 'Supervised Learning',
        'model_category': 'Generalized Linear Models',
        'model': 'LassoLars',
        'description': 'LassoLars is a lasso model implemented using the LARS algorithm, and unlike the implementation based on coordinate_descent, this yields the exact solution, which is piecewise linear as a function of the norm of its coefficients.',
        'tags': ['supervised', 'LassoLars', 'regression'],
        'connection': {
          'istarget': True,
          'issource': False,
          'maxtargets': 3,
          'maxsources': 0
        }
    },
    {
        'id': 6,
        'learning_type': 'Supervised Learning',
        'model_category': 'Generalized Linear Models',
        'model': 'BayesianRidge',
        'description': 'Bayesian regression techniques can be used to include regularization parameters in the estimation procedure: the regularization parameter is not set in a hard sense but tuned to the data at hand. BayesianRidge estimates a probabilistic model of the regression problem. ',
        'tags': ['supervised', 'bayesian', 'ridge', 'regression'],
        'connection': {
          'istarget': True,
          'issource': False,
          'maxtargets': 3,
          'maxsources': 0
        }
    },
    {
        'id': 7,
        'learning_type': 'Supervised Learning',
        'model_category': 'Stochastic Gradient Descent',
        'model': 'SGDRegressor',
        'description': 'SGD stands for Stochastic Gradient Descent: the gradient of the loss is estimated each sample at a time and the model is updated along the way with a decreasing strength schedule (aka learning rate). The regularizer is a penalty added to the loss function that shrinks model parameters towards the zero vector using either the squared euclidean norm L2 or the absolute norm L1 or a combination of both (Elastic Net). If the parameter update crosses the 0.0 value because of the regularizer, the update is truncated to 0.0 to allow for learning sparse models and achieve online feature selection. This implementation works with data represented as dense numpy arrays of floating point values for the features. The class SGDRegressor implements a plain stochastic gradient descent learning routine which supports different loss functions and penalties to fit linear regression models. SGDRegressor is well suited for regression problems with a large number of training samples (> 10.000), for other problems we recommend Ridge, Lasso, or ElasticNet.',
        'tags': ['supervised', 'stochastic', 'gradient', 'regression'],
        'connection': {
          'istarget': True,
          'issource': False,
          'maxtargets': 3,
          'maxsources': 0
        }
    },
    {
        'id': 8,
        'learning_type': 'Supervised Learning',
        'model_category': 'Stochastic Gradient Descent',
        'model': 'SGDClassifier',
        'description': 'SGD stands for Stochastic Gradient Descent: the gradient of the loss is estimated each sample at a time and the model is updated along the way with a decreasing strength schedule (aka learning rate). The regularizer is a penalty added to the loss function that shrinks model parameters towards the zero vector using either the squared euclidean norm L2 or the absolute norm L1 or a combination of both (Elastic Net). If the parameter update crosses the 0.0 value because of the regularizer, the update is truncated to 0.0 to allow for learning sparse models and achieve online feature selection. This implementation works with data represented as dense numpy arrays of floating point values for the features. The class SGDClassifier implements a plain stochastic gradient descent learning routine which supports different loss functions and penalties for classification.',
        'tags': ['supervised', 'stochastic', 'gradient', 'classifier', 'classification'],
        'connection': {
          'istarget': True,
          'issource': False,
          'maxtargets': 3,
          'maxsources': 0
        }
    },

    {
        'id': 9,
        'learning_type': 'Supervised Learning',
        'model_category': 'Support Vector Machines',
        'model': 'SVM',
        'description': 'Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.',
        'tags': ['supervised', 'support vector machine', 'classifier', 'classification'],
        'connection': {
          'istarget': True,
          'issource': False,
          'maxtargets': 3,
          'maxsources': 0
        }
    },
    {
        'id': 10,
        'learning_type': 'Supervised Learning',
        'model_category': 'Support Vector Regression',
        'model': 'SVR',
        'description': 'The method of Support Vector Classification can be extended to solve regression problems. This method is called Support Vector Regression.',
        'tags': ['supervised', 'support vector regression', 'regression'],
        'connection': {
          'istarget': True,
          'issource': False,
          'maxtargets': 3,
          'maxsources': 0
        }
    }
]
