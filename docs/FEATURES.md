# Feature contract and availability

All deployed inputs describe a **completed or observed aggregate session snapshot**. Although some counts can be updated during browsing, this dataset does not contain event sequences, snapshot times or outcome cutoffs. No feature is validated for prospectively predicting conversion early in a session.

| Source feature | Meaning and availability | Treatment |
|---|---|---|
| Administrative | Administrative page count at snapshot; final aggregate in UCI | Nonnegative integer; scaled numeric |
| Administrative_Duration | Aggregate time on administrative pages | Nonnegative duration; scaled numeric |
| Informational | Informational page count at snapshot | Nonnegative integer; scaled numeric |
| Informational_Duration | Aggregate informational-page time | Nonnegative duration; scaled numeric |
| ProductRelated | Product-page count at snapshot | Nonnegative integer; scaled numeric |
| ProductRelated_Duration | Aggregate product-page time | Nonnegative duration; scaled numeric |
| BounceRates | Session-associated page bounce metric from analytics | Proportion [0,1]; point-in-time source availability must be audited |
| ExitRates | Session-associated page exit metric from analytics | Proportion [0,1]; may require completed page/session analytics |
| PageValues | Page value associated with transactions | **Excluded**, potential transaction-associated leakage |
| SpecialDay | Calendar-based proximity to a special day | Proportion [0,1]; potentially known in advance |
| Month | Session month | One-hot plus optional cyclical encoding; UCI uses `June` |
| OperatingSystems | Anonymous operating-system category code | Positive integer encoded categorically |
| Browser | Anonymous browser category code | Positive integer encoded categorically |
| Region | Anonymous region category code | Positive integer encoded categorically |
| TrafficType | Anonymous source category code | Positive integer encoded categorically; no invented channel names |
| VisitorType | New, returning or other visitor classification | One-hot; assumes visitor identity history is available |
| Weekend | Session calendar indicator | Boolean categorical; known at visit time |
| Revenue | Whether the session purchased | **Target only**, available after outcome resolution |

Unseen positive device/region/source codes are accepted and become an unknown category in the encoder. Unknown month and visitor-type labels are rejected as likely schema errors. No arbitrary upper bound is invented for counts or duration; IQR flags describe extremes without clipping them. Rates are bounded by the semantic range [0,1], not the empirical UCI maximum.

## Engineered feature set

`TotalPages` sums the three page counts; `TotalDuration` sums the three duration fields. `DurationPerPage` divides those totals. `ProductPageRatio` and `ProductDurationRatio` measure the product fraction of total activity. `ProductTimePerPage` measures product time per product page. `BounceExitRatio` divides bounce by exit rate. `MonthSin` and `MonthCos` provide annual cyclical position.

Every ratio returns zero when its denominator is zero. These are deterministic transformations with no fitted population statistics. Some derived features are correlated with their inputs; they are retained only as a candidate set in the ablation. Explicit returning/weekend copies and all complementary engagement ratios are not added because their information is already present. The feature-screening candidate ranks train-only mutual information, drops constants, prunes absolute correlations at 0.95 and keeps at most 80% of encoded columns. Cross-validation refits this selector inside each fold.

`Revenue` and `PageValues` are dropped before preprocessing even if a caller supplies a full dataset dataframe. API requests exclude both fields entirely. Imputers are fitted within the training pipeline as defensive support for controlled experiments, but deployed input validation rejects missing values rather than silently guessing an incomplete session.
