# Data Quality and Validation Report: YouTube Spam Collection

## 1. Executive Summary
- **Dataset Citation**: Alberto, T. & Lochter, J. (2015). YouTube Spam Collection. UCI Machine Learning Repository. [https://doi.org/10.24432/C58885](https://doi.org/10.24432/C58885)
- **Total Rows Loaded**: 1956
- **Total Unique Rows After Exact-Duplicate Removal**: 1953
- **Exact Duplicate Rows Removed**: 3
- **Empty or Whitespace-Only Comments**: 0
- **Cleaned Dataset Location**: `data/processed/youtube_spam_cleaned.csv`

---

## 2. Overall Class Distribution
| Class | Label ID | Count | Percentage |
| :--- | :---: | :---: | :---: |
| Spam | 1 | 1003 | 51.36% |
| Ham (Legitimate) | 0 | 950 | 48.64% |
| **Total** | - | **1953** | **100.0%** |

---

## 3. Per-Source Video Client Distribution
| Video Subset (Client) | Total Rows | Spam (1) | Ham (0) | Missing Dates | Missing Authors |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Psy | 350 | 175 | 175 | 0 | 0 |
| KatyPerry | 350 | 175 | 175 | 0 | 0 |
| LMFAO | 438 | 236 | 202 | 0 | 0 |
| Eminem | 448 | 245 | 203 | 245 | 0 |
| Shakira | 370 | 174 | 196 | 0 | 0 |


---

## 4. Duplicate Records Audit
The UCI dataset contains 1,956 raw records across five files.
Exact duplicates (where all attributes match identically) were identified and logged prior to removal:

| Video | Comment ID | Author | Label | Text Snippet |
|---|---|---|---|---|
| Eminem | `LneaDw26bFvPh9xBHNw1btQoyP60ay_WWthtvXCx37s` | janez novak | 1 | share and like this page to win a hand signed Riha... |
| Eminem | `LneaDw26bFuH6iFsSrjlJLJIX3qD4R8-emuZ-aGUj0o` | Amir bassem | 1 | if u love rihanna subscribe me... |
| Shakira | `_2viQ_Qnc68fX3dYsfYuM-m4ELMJvxOQBmBOFHqGOk0` | tyler sleetway | 0 | so beutiful... |


### Explanation of Row Count Divergence
The original UCI repository summary mentions 1,956 comments. Upon auditing each individual file:
- Psy: 350
- Katy Perry: 350
- LMFAO: 438
- Eminem: 448 (includes 2 exact duplicates)
- Shakira: 370 (includes 1 exact duplicate)
Summing raw rows: `350 + 350 + 438 + 448 + 370 = 1,956`.
Removing the 3 exact duplicate entries results in **1,953** unique records for training and evaluation.

---

## 5. Schema and Column Specifications
| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `comment_id` | `str` |
| `author` | `str` |
| `date` | `str` |
| `raw_content` | `str` |
| `original_class` | `int64` |
| `video_source` | `str` |
| `source_file` | `str` |
| `label` | `int64` |
| `label_name` | `str` |
| `processed_content` | `str` |


---

## 6. Target Mapping Verification
Inspection of ground truth examples verifies that:
- `CLASS = 1` corresponds strictly to **Spam** (containing promotional URLs, subscription requests, external website links).
- `CLASS = 0` corresponds strictly to **Ham** (organic viewer reactions to the song, artist, and video).
