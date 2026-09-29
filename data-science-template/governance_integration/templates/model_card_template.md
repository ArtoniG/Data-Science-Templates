# Regulatory Model Card: {{ model_name }}
**Version:** {{ model_version }} | **Date Generated:** {{ generation_date }}
**Primary Owner:** {{ model_owner }} | **Status:** {{ deployment_status }}

---

## 1. Model Overview & Intended Use

**Model Description:** 
{{ model_description }}

**Intended Use Cases:**
* Automated application scoring for unsecured retail credit products.
* Risk-based pricing and initial credit line assignment.

**Out-of-Scope Use Cases:**
* Mortgage lending, small business lending, or automated account management (line decreases/closures).

**Regulatory Frameworks Assessed:**
* Fair Credit Reporting Act (FCRA) / Adverse Action Code Generation.
* Equal Credit Opportunity Act (ECOA) / Fair Lending.
* Internal Model Risk Management (MRM) Policies.

---

## 2. Architecture & Serialization

**Algorithmology:**
* **Imputation Strategy:** {{ imputation_strategy }}
* **Encoding:** Weight of Evidence (WOE) with strict monotonicity enforcement.
* **Classifier:** L2-Regularized Logistic Regression (C={{ logistic_c }}).
* **Base Score & PDO:** Base {{ base_score }} at {{ odds_ratio }} odds, Points to Double Odds (PDO) = {{ pdo }}.

**Deployment Mechanism:**
This model completely bypasses native binary serialization (e.g., `pickle`, `joblib`). It is strictly reconstructed in memory using a deterministic, declarative `pipeline_state.json` artifact, eliminating deserialization vulnerabilities.

---

## 3. Data Lineage & Target Definition

**Training Data Scope:**
* **Observation Window:** {{ train_start_date }} to {{ train_end_date }}
* **Total Records (Train):** {{ train_size }}
* **Total Records (Test):** {{ test_size }}

**Target Definition:**
* **Bad Definition (1):** {{ target_definition_bad }}
* **Good Definition (0):** {{ target_definition_good }}
* **Indeterminate (Excluded):** {{ target_definition_indeterminate }}

**Protected Classes Excluded:**
Age, Race, Gender, Marital Status, Religion, National Origin.

---

## 4. Performance & Discrimination Metrics

| Metric | Training Set | Validation Set | Minimum Threshold | Status |
|---|---|---|---|---|
| **Gini Coefficient** | {{ gini_train }} | {{ gini_test }} | > 0.40 | {{ status_gini }} |
| **Kolmogorov-Smirnov (KS)** | {{ ks_train }} | {{ ks_test }} | > 0.35 | {{ status_ks }} |
| **ROC AUC** | {{ auc_train }} | {{ auc_test }} | > 0.70 | {{ status_auc }} |

> **Audit Note:** The Gini divergence between Training and Validation sets is {{ gini_divergence }}%. A divergence exceeding 10% indicates potential overfitting and requires MRM review.

---

## 5. Feature Information Value (IV) & Stability

The following table details the primary predictive features, their discriminative power (IV), and the established baseline Population Stability Index (PSI) used for production monitoring.

| Feature Name | Type | Information Value (IV) | PSI Baseline Bins | Monotonicity Verified |
|---|---|---|---|---|
{{ feature_metrics_table_rows }}

---

## 6. Adverse Action (Reason Codes) Mapping

The model integrates directly with the `CreditScoringEngine` and `ReasonCodeExtractor` to provide legally mandated adverse action explanations. The top drivers for score penalization are dynamically calculated per applicant based on WOE coefficient deviation.

**Top Global Penalty Features:**
1. {{ top_reason_1 }}
2. {{ top_reason_2 }}
3. {{ top_reason_3 }}

---

## 7. Model Risk & Compliance Sign-off

By approving this document, the reviewing officer confirms that the `pipeline_state.json` artifact has passed all automated Quality Gates (Phase 6), including feature monotonicity, NaN handling, and absence of severe collinearity.

* **Data Science Lead:** _________________________ Date: _________
* **Model Risk Mgmt (MRM):** _________________________ Date: _________
* **Compliance / Legal:** _________________________ Date: _________