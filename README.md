Credit Scoring Business Understanding

●	How does the Basel II Accord's emphasis on risk measurement influence the need for an interpretable and well-documented model?

    1.	Basel II Compliance & Interpretability: Basel II demands strict capital adequacy assessments based on risk. Because credit decisions affect financial stability, black-box models aren't legally viable. Regulators and risk teams must be able to audit exactly why an applicant was denied or granted a specific credit limit.


●	Without a direct "default" label, why is a proxy variable necessary, and what business risks does proxy-based prediction introduce?

    2.	The Proxy Variable Dilemma: Because the Xente dataset lacks a traditional "defaulted (1) vs. Paid (0)" column, you must engineer a proxy (using RFM clustering to isolate unengaged/low-monetary accounts). The Business Risk: You might misclassify a safe, new user as "high risk" (False Positive - lost revenue) or a highly active fraudulent user as "low risk" (False Negative - bad debt).


●	What are the key trade-offs between a simple, interpretable model (e.g., Logistic Regression with WoE) and a high-performance model (e.g., Gradient Boosting) in a regulated financial context?

    3.	Model Trade-offs: * Logistic Regression + WoE: Highly interpretable, satisfies regulators easily, stable, but might miss non-linear feature interactions.
    o	Gradient Boosting (XGBoost/LightGBM): High predictive accuracy, handles complex patterns well, but behaves as a black box requiring complex explainability tools (like SHAP) to satisfy Basel II.

Metrices in simple terms:
•	Accuracy: how often the model is correct overall
•	Precision: when the model says “yes,” how often it is right
•	Recall: out of all the real “yes” cases, how many it finds
•	F1 score: a balance between precision and recall
Quick example: Imagine spam detection.
•	Accuracy: how many emails were classified correctly total
•	Precision: of the emails marked as spam, how many were actually spam
•	Recall: of all actual spam emails, how many the model caught
•	F1: useful when you want both precision and recall to be good
