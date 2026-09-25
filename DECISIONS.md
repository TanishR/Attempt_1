# Project Decisions & EDA Rationale

Ground truth analysis and architectural decisions for the Amazon ML Challenge 2026 Entity Resolution pipeline.

---

## 1. EDA Findings & Design Parameters (Guidebook Step 2.5)

Based on comprehensive Exploratory Data Analysis over the full training set (2.2M train S1, 5.0M train S2, 5.3M train S3):

| Metric / Check | Value / Finding | Decision / Parameter Setting | Rationale |
| --- | --- | --- | --- |
| **Exclusivity Violations** | **0** (0 S2/S3 IDs match >1 S1) | `USE_EXCLUSIVITY = True` | In the ground truth, every S2/S3 entity belongs exclusively to at most one S1 entity. Applying an exclusivity filter in Step 8 (assigning shared candidate to the highest probability S1) eliminates contradictory duplicates and optimizes precision. |
| **Cross-Country Matches** | **0** (0 cross-border matches) | `BLOCK_BY_COUNTRY = True` | Entities never match across different countries. Partitioning candidate generation strictly by country (US, India, France) is 100% sound, avoids false cross-border matches, and drastically reduces GPU matmul and CPU merge sizes. (Country is used for blocking only, never as a model feature). |
| **Singleton Percentage** | **5.58%** (US: 5.58%, India: 5.59%) | Moderate `t_top1` threshold | Only ~5.6% of S1 entities have no matches in S2/S3. Because singletons are rare and missing a true match yields a 0 score, `t_top1` should start moderately (0.30–0.50) to avoid false singleton predictions. |
| **Matches per S1** | S2: 95th pct = 4, max = 5<br>S3: 95th pct = 4, max = 6<br>Total: 95th pct = 6, max = 11 | `K_PER_SOURCE = 10`<br>`CAND_CAP = 30` | 95% of entities have $\le 4$ matches per source and $\le 6$ matches total. Setting `K_PER_SOURCE = 10` provides 2.5x coverage over the 95th percentile, and `CAND_CAP = 30` comfortably accommodates multi-match clusters while capping feature computation and memory usage. |

---

## 2. Architectural Deviation: Dual Name Embeddings for Non-ASCII Records

### Specification
* **ASCII records:** Embed normalized `name_full` only $\rightarrow$ stored in `emb_<split>_<source>.npy` (fp16, 256d).
* **Non-ASCII records:**
  1. **Main Embedding:** Embed `raw_name` (preserving native UTF-8 script) $\rightarrow$ stored in `emb_<split>_<source>.npy`.
  2. **Alternative Embedding:** Embed normalized `name_full` (`anyascii` transliterated) $\rightarrow$ stored in `emb_<split>_<source>_alt.npy`.

### Rationale
1. **Multilingual Model Capacity:** `Qwen/Qwen3-Embedding-0.6B` is pre-trained natively on extensive multilingual corpora, including Indic scripts (Devanagari, Tamil, Telugu, Kannada, Gujarati, Bengali) and accented Latin (French). It produces high-quality representations directly from native scripts.
2. **Information Loss in Transliteration:** Pure rule-based transliteration (`anyascii`) loses native phonological subtleties and script distinctions, potentially conflating distinct words.
3. **Cross-Script Ground Truth Pairs:** In the dataset, ground-truth matches occur across scripts (e.g., Devanagari in S2/S3 matched to Latin in S1, or native script in both). Generating both raw script and transliterated embeddings allows multi-vector max-pooling in Channel A search:
   $$\text{score}(S_1, C_2) = \max\Big(Q_{\text{main}} \cdot X_{\text{main}}^T, \, Q_{\text{main}} \cdot X_{\text{alt}}^T, \, Q_{\text{alt}} \cdot X_{\text{main}}^T, \, Q_{\text{alt}} \cdot X_{\text{alt}}^T\Big)$$
   This guarantees maximal recall regardless of whether the pair is native-to-native, native-to-Latin, or Latin-to-Latin.
