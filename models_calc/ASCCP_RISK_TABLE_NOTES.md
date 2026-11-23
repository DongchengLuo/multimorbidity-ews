# ASCCP Official Risk Table — Implementation Notes

## Data sources

- Original file: `ASCCP Risk.pdf` (cervical cancer guideline folder)
- Reference: Egemen D, et al. *Risk Estimates Supporting the 2019 ASCCP Risk-Based Management Consensus Guidelines*. J Low Genit Tract Dis. 2020;24:132-143.
- Implementation version: `2019-ASCCP-Egemen-2020`
- Structured data: `asccp_risk_table.csv`
- Partial genotyping source: Demarco M, et al. *A Study of Partial Human Papillomavirus Genotyping in Support of the 2019 ASCCP Risk-Based Management Consensus Guidelines*. J Low Genit Tract Dis. 2020;24:144–147. DOI: `10.1097/LGT.0000000000000530`
- Partial genotyping data: `asccp_partial_genotyping_risk_table.csv`
- Lookup engine: `asccp_risk_engine.py`

Risk values were entered row by row from the original Tables 1A–5B, covering 101 official combinations. The program performs exact table lookup only — no interpolation, and no back-derivation of risk values not provided in the source. The original management column in the tables is used only for formula/table verification; upstream model output does not pass management recommendations to downstream reports. Report recommendations are generated uniformly by the downstream LLM stage.

Partial genotyping is additionally entered from Demarco Table 1 for the 18 combinations of HPV16, HPV18, HR12 and 6 cytology results. This table is also a direct lookup, not a formula. Output contains only immediate/5-year CIN3+ risk, risk stratification, code and source information; it does not output the original management recommendations or special rules.

## Scenarios and inputs

| Scenario code | Source table | Required inputs |
|---|---|---|
| `initial_unknown` | 1A; Demarco Table 1 when genotyping is available | Current HPV, current cytology; genotyping may be HPV16, HPV18 or the other 12 high-risk types (HR12) |
| `initial_prior_hpv_negative` | 1B | Current HPV, current cytology; prior negative must be a documented HPV test or co-test within ~5 years |
| `surveillance_hpv_negative_ascus` | 2A | Current HPV, current cytology |
| `surveillance_hpv_negative_lsil` | 2B | Current HPV, current cytology; some long-term negative combinations require the number of negatives and test type |
| `surveillance_hpv_positive_nilm` | 2C | Current HPV, current cytology; some long-term negative combinations require the number of negatives and test type |
| `colposcopy_biopsy_result` | 3 | Pre-colposcopy screening result, biopsy pathology |
| `postcolposcopy_low_grade` | 4A | Current HPV, current cytology |
| `postcolposcopy_high_grade` | 4B | Current HPV, current cytology |
| `post_treatment_first_followup` | 5A | Current HPV, current cytology |
| `post_treatment_long_term` | 5B | Number of consecutive negative follow-ups before this test, current negative test type |

For Table 5B, "number of consecutive negative follow-ups" means the number of negatives already recorded before the current negative result: 1 means the current result is the 2nd negative, 2 means it is the 3rd negative.

## Strict limitations

- The source risk cohort is 25–65 years old. The table is not applied below 25; the engine returns "special population not covered".
- Pregnancy and immunosuppressed populations are not covered by the general risk table; the engine returns a dedicated rule notice.
- Tables 1A–5B use the combined positive/negative result of 13 high-risk HPV types. When HPV16, HPV18 or HR12 is explicitly known, the `initial_unknown` scenario switches to Demarco Table 1 and does not use the combined risk value.
- Demarco Table 1 covers only "no known prior HPV testing history". When prior HPV history, post-colposcopy or post-treatment scenarios exist, genotyping input is not applied to this table and the engine explicitly returns not-covered.
- "HPV16/18 combined positive" cannot determine whether it is HPV16 or HPV18; the program does not guess the group. A distinguishable specific type must be provided. The three source tiers are HPV16, HPV18 and HR12.
- Special combinations without reliable source values keep an empty risk value and output the source special-handling recommendation; percentages are not fabricated.
- When no official row can be uniquely matched, the engine returns "not covered" and does not perform approximate matching.

## Table self-checks

`validate_table()` checks the total row count of the combined risk table, per-table row counts, duplicate lookup keys, risk-value ranges, validity of management codes, and source page/version. `validate_partial_genotyping_table()` checks that the 18 combinations of the genotyping table are complete, unique and have valid risk values.
