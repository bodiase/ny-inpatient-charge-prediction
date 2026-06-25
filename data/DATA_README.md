# Data

## Source

New York State Department of Health  
**Hospital Inpatient Discharges (SPARCS De-Identified) — 2021**  
https://health.data.ny.gov/Health/Hospital-Inpatient-Discharges-SPARCS-De-Identified/gnzp-ekau

SPARCS (Statewide Planning and Research Cooperative System) is a comprehensive data reporting system that collects patient-level data on hospital discharges in New York State. The 2021 dataset covers all inpatient hospitalizations statewide.

## Files in this folder

**`sample_sparcs_2021_500rows.csv`**  
A 500-row stratified sample of the 50,000-record working dataset, included for local reproducibility. Stratified by APR Severity of Illness and primary payer type to preserve the distribution of key variables.

This sample is sufficient to verify the full pipeline runs correctly. For the complete analysis, the notebook loads all 50,000 records directly from Google Drive (URL in the data loading cell).

## Privacy and Licensing

SPARCS data is de-identified per HIPAA Safe Harbor standards before public release by the New York State Department of Health. No individually identifiable health information is present in the published fields. The data is publicly available and free to use for research and analysis.

## Full Dataset

The full 50,000-record sample used in this analysis is available at:  
https://docs.google.com/uc?export=download&id=1XQ1-fFmE1XmRQ1G3f3OPvktl8nXOHW5_

The complete 2021 SPARCS statewide census (all discharge records) is available from the NY DOH portal linked above.
