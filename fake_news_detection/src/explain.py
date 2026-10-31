"""Stage 5 - LIME explainability layer (word-level)."""
from lime.lime_text import LimeTextExplainer


def explain(pipe, text, num_features=10, num_samples=500):
    """Returns [(word, weight)]: weight > 0 pushes toward FAKE, weight < 0 toward REAL."""
    explainer = LimeTextExplainer(class_names=["Real", "Fake"], random_state=42)
    exp = explainer.explain_instance(text, pipe.predict_proba, labels=(1,),
                                     num_features=num_features, num_samples=num_samples)
    return [(str(w), float(v)) for w, v in exp.as_list(label=1)]
