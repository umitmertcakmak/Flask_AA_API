transformers = [
    {
        'id': 1,
        'category': 'Feature Extractors',
        'transformer': 'Dictionary Vectorizer',
        'description': 'The class DictVectorizer can be used to convert feature arrays represented as lists of standard Python dict objects to the NumPy/SciPy representation used by scikit-learn estimators.',
        'tags': ['transformer'],
        'connection': {
          'istarget': True,
          'issource': True,
          'maxtargets': 3,
          'maxsources': 3
        }
    },
    {
        'id': 2,
        'category': 'Feature Extractors',
        'transformer': 'Count Vectorizer',
        'description': 'Count Vectorizer implements both tokenization and occurrence counting in a single class.',
        'tags': ['transformer'],
        'connection': {
          'istarget': True,
          'issource': True,
          'maxtargets': 3,
          'maxsources': 3
        }
    },
    {
        'id': 3,
        'category': 'Feature Extractors',
        'transformer': 'Tfidf Transformer',
        'description': 'Tf means term-frequency while tf–idf means term-frequency times inverse document-frequency.',
        'tags': ['transformer'],
        'connection': {
            'istarget': True,
            'issource': True,
            'maxtargets': 3,
            'maxsources': 3
        }
    },
    {
        'id': 4,
        'category': 'Feature Extractors',
        'transformer': 'Hashing Vectorizer',
        'description': 'Convert a collection of text documents to a matrix of token occurrences.',
        'tags': ['transformer'],
        'connection': {
            'istarget': True,
            'issource': True,
            'maxtargets': 3,
            'maxsources': 3
        }
    },
    {
        'id': 5,
        'category': 'Preprocessing',
        'transformer': 'Scale',
        'description': 'Standardization of datasets is a common requirement for many machine learning estimators implemented in scikit-learn; they might behave badly if the individual features do not more or less look like standard normally distributed data: Gaussian with zero mean and unit variance.',
        'tags': ['transformer'],
        'connection': {
            'istarget': True,
            'issource': True,
            'maxtargets': 3,
            'maxsources': 3
        }
    },
    {
        'id': 6,
        'category': 'Preprocessing',
        'transformer': 'Standard Scaler',
        'description': 'Standardize features by removing the mean and scaling to unit variance.',
        'tags': ['transformer'],
        'connection': {
            'istarget': True,
            'issource': True,
            'maxtargets': 3,
            'maxsources': 3
        }
    },
    {
        'id': 7,
        'category': 'Preprocessing',
        'transformer': 'MinMax Scaler',
        'description': 'Transforms features by scaling each feature to a given range.',
        'tags': ['transformer'],
        'connection': {
            'istarget': True,
            'issource': True,
            'maxtargets': 3,
            'maxsources': 3
        }
    },
    {
        'id': 8,
        'category': 'Preprocessing',
        'transformer': 'MinAbs Scaler',
        'description': 'Scale each feature by its maximum absolute value.',
        'tags': ['transformer'],
        'connection': {
            'istarget': True,
            'issource': True,
            'maxtargets': 3,
            'maxsources': 3
        }
    },
    {
        'id': 9,
        'category': 'Preprocessing',
        'transformer': 'Robust Scaler',
        'description': 'Scale features using statistics that are robust to outliers.',
        'tags': ['transformer'],
        'connection': {
            'istarget': True,
            'issource': True,
            'maxtargets': 3,
            'maxsources': 3
        }
    },
    {
        'id': 10,
        'category': 'Preprocessing',
        'transformer': 'Kernel Centerer',
        'description': 'Center a kernel matrix.',
        'tags': ['transformer'],
        'connection': {
            'istarget': True,
            'issource': True,
            'maxtargets': 3,
            'maxsources': 3
        }
    }
]
