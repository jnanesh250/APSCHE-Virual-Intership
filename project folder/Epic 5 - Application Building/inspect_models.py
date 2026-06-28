import joblib
import os

base = os.path.dirname(__file__)
models = [
    os.path.normpath(os.path.join(base, '..', 'Epic 4 - Model Building', 'rf_model.joblib')),
    os.path.normpath(os.path.join(base, '..', 'Epic 4 - Model Building', 'dt_model.joblib')),
    os.path.normpath(os.path.join(base, '..', 'Epic 4 - Model Building', 'logistic_model.joblib')),
]

for path in models:
    print('\nMODEL:', path)
    if not os.path.exists(path):
        print('  not found')
        continue
    try:
        m = joblib.load(path)
        def find_feature_names(obj, visited=None):
            if visited is None:
                visited = set()
            if id(obj) in visited:
                return None
            visited.add(id(obj))
            if hasattr(obj, 'feature_names_in_'):
                try:
                    return list(obj.feature_names_in_)
                except Exception:
                    return None
            # sklearn Pipeline
            if hasattr(obj, 'named_steps'):
                for step in obj.named_steps.values():
                    fn = find_feature_names(step, visited)
                    if fn:
                        return fn
            if hasattr(obj, 'steps'):
                for _, step in obj.steps:
                    fn = find_feature_names(step, visited)
                    if fn:
                        return fn
            # estimator inside meta-estimators
            if hasattr(obj, 'estimator_'):
                return find_feature_names(obj.estimator_, visited)
            return None

        fn = find_feature_names(m)
        print('  has_feature_names:', bool(fn))
        if fn is not None:
            print('  feature_names:', fn)
        else:
            print('  estimator class:', type(m))
    except Exception as e:
        print('  ERROR loading model:', e)
