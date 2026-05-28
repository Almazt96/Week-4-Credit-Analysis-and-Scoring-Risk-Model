Credit Scoring Business Understanding

●	How does the Basel II Accord's emphasis on risk measurement influence the need for an interpretable and well-documented model?
●	Without a direct "default" label, why is a proxy variable necessary, and what business risks does proxy-based prediction introduce?
●	What are the key trade-offs between a simple, interpretable model (e.g., Logistic Regression with WoE) and a high-performance model (e.g., Gradient Boosting) in a regulated financial context?

Answers

1.	Basel II Compliance & Interpretability: Basel II demands strict capital adequacy assessments based on risk. Because credit decisions affect financial stability, black-box models aren't legally viable. Regulators and risk teams must be able to audit exactly why an applicant was denied or granted a specific credit limit.
2.	The Proxy Variable Dilemma: Because the Xente dataset lacks a traditional "defaulted (1) vs. Paid (0)" column, you must engineer a proxy (using RFM clustering to isolate unengaged/low-monetary accounts). The Business Risk: You might misclassify a safe, new user as "high risk" (False Positive - lost revenue) or a highly active fraudulent user as "low risk" (False Negative - bad debt).
3.	Model Trade-offs: * Logistic Regression + WoE: Highly interpretable, satisfies regulators easily, stable, but might miss non-linear feature interactions.
o	Gradient Boosting (XGBoost/LightGBM): High predictive accuracy, handles complex patterns well, but behaves as a black box requiring complex explainability tools (like SHAP) to satisfy Basel II.
