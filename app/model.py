import joblib
import numpy as np
from pathlib import Path
import pandas as pd

MODELS_DIR = Path(__file__).parent.parent / "models"

scaler = joblib.load(MODELS_DIR / "scaler.joblib")
kmeans = joblib.load(MODELS_DIR / "kmeans_model.joblib")

CLUSTER_NAMES = {
    0: "Champions",
    1: "At Risk",
    2: "Loyal",
    3: "Hibernating"
}

STRATEGIES = {
    "Champions": "Reward and retain — enroll in VIP loyalty program, no discounts needed.",
    "Loyal": "Upsell and graduate — offer bundles and tiered loyalty points.",
    "At Risk": "Reactivation — send win-back email with 15% discount and 2 week expiry.",
    "Hibernating": "Minimal spend — one final offer, suppress if no response."
}

TOP_ACTIONS = {
    "Champions": "Enroll in VIP program and request referrals.",
    "Loyal": "Offer bundle deals to increase average order value.",
    "At Risk": "Send win-back email with limited time discount.",
    "Hibernating": "One final reactivation email, then suppress from lists."
}




FEATURE_NAMES = ['Recency', 'Frequency', 'Monetary', 'AvgOrderValue', 'CustomerLifetime']

def predict_segment(recency, frequency, monetary, avg_order_value, customer_lifetime):
    features = np.array([[recency, frequency, monetary, avg_order_value, customer_lifetime]])

    caps = [369.0, 30.0, 19881.0, 2031.16, 367.0]
    for i in range(features.shape[1]):
        features[0, i] = min(features[0, i], caps[i])

    features_log = np.log1p(features)
    features_df = pd.DataFrame(features_log, columns=FEATURE_NAMES)

    features_scaled = scaler.transform(features_df)
    features_scaled_df = pd.DataFrame(features_scaled, columns=FEATURE_NAMES)

    cluster_id = int(kmeans.predict(features_scaled_df)[0])
    segment = CLUSTER_NAMES[cluster_id]

    return {
        "cluster_id": cluster_id,
        "segment": segment,
        "strategy": STRATEGIES[segment],
        "top_action": TOP_ACTIONS[segment]
    }