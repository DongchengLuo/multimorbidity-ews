#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_snapshot.py - generate a fully de-identified / anonymized local demo snapshot (5000 records)

Design principles
--------
1. No database connection. All input comes from existing historical CSVs under batches/ in this repo.
2. No real patient's full record is copied:
   - Lab values: learn only each item's distribution from historical data (quantiles / category frequencies),
     then resample randomly from that distribution, so no row maps one-to-one to a real patient.
   - Descriptive text: uses the template library in this file; no real report text is quoted.
3. All direct identifiers are fabricated with no mapping to real values:
   - app_no   : 90000001 ~ 90005000 (8 digits, original format kept, clearly a demo range)
   - pat_id   : P900000001 ~ P900005000
   - Name     : Subject0001 ~ Subject5000
   - Address  : No. N, Virtual Street, Demo District, Demo City, Demo Province
   - Ethnicity: Han (constant, carries no information)
   - ID number / phone / email / visit card / contacts / photo etc.: all left blank
4. Output directory data/snapshot/ with 6 CSVs mirroring the real database.

Usage
----
    python tools/build_snapshot.py
    python tools/build_snapshot.py --count 5000 --seed 20261110
"""

import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BATCH_GLOB = os.path.join(ROOT, "batches", "*", "PES_RESULT.csv")
OUT_DIR = os.path.join(ROOT, "data", "snapshot")

# ---------------------------------------------------------------- Constants

APP_NO_START = 90000001
PAT_ID_PREFIX = "P900"
DETAIL_ID_START = 90000000
ADDRESS_TMPL = "No. {n} Virtual Street, Demo District, Demo City, Demo Province"
NATION = "Han"
NATIONALITY = "chinese"
DOCTOR_NAME = "Demo Physician"

# Real table structure (column order matches production; downstream base_select / desc_item rely on names)
COLS_APPLY = [
    "app_no", "app_org_id", "app_time", "pat_id", "app_pat_name", "app_pat_age",
    "app_his_visitno", "app_loc_id", "app_pkg_id", "app_pkg_name", "app_extype_id",
    "app_flag", "app_co_id", "app_co_batch", "app_co_group", "app_co_depart",
    "app_signin_time", "app_signin_flag", "app_signout_time", "app_signout_flag",
    "app_seq", "app_bed_no", "app_userid", "app_username", "app_price",
    "app_guide_printflag", "app_rep_printflag", "app_rep_printtime", "app_del_flag",
    "app_his_visittime", "app_workno", "app_check_date", "app_pat_agemonth",
    "app_rep_takemode", "app_invoice_title", "app_remark", "app_co_name",
    "app_position", "pat_id_origin", "app_terminal", "app_no_origin", "app_lock",
    "app_rep_printusername", "app_manager_doctor", "app_signin_userid",
    "app_signin_username", "app_source", "app_guide_printtime",
    "app_guide_printusername", "app_present", "app_level", "app_prefer_price",
    "app_receive_price", "app_signout_username", "app_balance_type",
    "app_check_times", "app_secrect", "app_slevel_id", "app_person_price",
    "app_group_price", "app_breakfast", "app_parent_no", "org_id",
    "his_visit_type", "app_rep_mode", "app_split_no",
]

COLS_PATIENT = [
    "pat_id", "pat_name", "pat_sex", "pat_birth_date", "pat_card_no",
    "pat_idcard_type", "pat_idcard", "pat_his_id", "pat_his_id2", "pat_family_no",
    "pat_custom_id", "pat_nationality", "pat_nation", "pat_nativeplace",
    "pat_address", "pat_email", "pat_mobile", "pat_marriage", "pat_job",
    "pat_contact", "pat_contact_mobile", "pat_idcardphoto", "pat_photo",
    "pat_del_flag", "pat_valid", "pat_valid_time", "pat_password",
    "pat_create_time", "pat_create_userid", "pat_create_username",
    "pat_lastupdate_time", "pycode", "pat_level_id", "pat_education",
    "pat_in_no", "pat_mailaddress_id", "pat_empi_id", "pat_passport",
    "pat_view_elecreport", "pat_login_flag", "pat_marriage_code",
]

COLS_RESULT = [
    "detail_id", "res_itm_id", "app_no", "res_com_id", "res_value", "res_value2",
    "res_unit", "res_refl", "res_refu", "res_info", "res_date", "res_seq",
    "res_exp", "res_value3", "res_value4", "res_itm_name", "res_specimen",
    "res_sum_flag", "res_value_num", "res_panic",
]

COLS_ITEM = [
    "detail_id", "det_time", "app_no", "pat_id", "det_com_id", "det_com_name",
    "det_dep_id", "det_dep_name", "det_price", "det_amount", "det_discount",
    "det_totalprice", "det_discount_totalprice", "det_charge_flag",
    "det_charge_type", "det_rep_type", "det_userid", "det_username",
    "det_usercode", "det_flag", "det_signin_flag", "det_signin_time",
    "det_append_flag", "det_aba_flag", "det_delay_flag", "det_apply_sendflag",
    "det_book_date", "det_book_optime", "det_book_bpid", "det_book_seq",
    "det_book_no", "det_ext_requestid", "det_ext_reportid", "det_lastupdate_time",
    "det_del_flag", "det_balance_flag", "det_restype", "det_add_item",
    "det_checkself", "det_fee_type", "det_apply_sendtime", "det_aba_time",
    "det_aba_username", "det_guide_printflag", "det_guide_printtime",
    "det_guide_printuser", "det_delay_time", "det_pkg_id", "det_remark",
    "det_delay_username", "det_parent_detailid", "det_gro_groupname",
    "det_extend_flag",
]

COLS_DESC = ["app_no", "detail_id", "res_result", "res_date"]

COLS_SUMMARY = [
    "detail_id", "app_no", "sum_summary", "sum_advice", "sum_advice_spec",
    "sum_create_time", "sum_create_userid", "sum_create_usercode",
    "sum_create_username", "sum_exam_time", "sum_exam_userid",
    "sum_exam_usercode", "sum_exam_username", "sum_audit_time",
    "sum_audit_userid", "sum_audit_usercode", "sum_audit_username",
]


# ---------------------------------------------------------------- Descriptive text template library
# Each exam item has several variants (normal / mildly abnormal / markedly abnormal), sampled by weight.
# The text is synthesized for this project and contains no real patient information.

DESC_TEMPLATES = {
    "ecg": [
        (0.55, "Sinus rhythm, heart rate 72 bpm; P-QRS-T morphology normal in all leads; no significant ST-segment deviation; upright T waves."),
        (0.25, "Sinus rhythm, heart rate 88 bpm; mild ST depression of 0.05 mV in leads II, III and aVF; flattened T waves; correlate with clinical findings."),
        (0.12, "Sinus bradycardia, heart rate 54 bpm; normal axis; no obvious ST-T changes."),
        (0.08, "Sinus rhythm with frequent premature ventricular contractions in bigeminy; cardiology follow-up advised."),
    ],
    "Holter ECG": [
        (0.6, "Sinus rhythm throughout, mean heart rate 71 bpm; no long pauses or significant dynamic ST-T changes."),
        (0.4, "Sinus rhythm throughout with 128 premature atrial contractions and 2 short runs of atrial tachycardia; cardiology evaluation advised."),
    ],
    "Ambulatory BP monitoring": [
        (0.5, "24-hour mean blood pressure 128/78 mmHg; dipper diurnal rhythm; blood pressure load normal."),
        (0.5, "24-hour mean blood pressure 146/92 mmHg; non-dipper diurnal rhythm; increased daytime blood pressure load."),
    ],
    "Thyroid ultrasound": [
        (0.42, "Thyroid normal in size and shape; homogeneous echogenicity in both lobes; no obvious mass. CDFI: normal intraglandular vascularity. No obviously enlarged cervical lymph nodes."),
        (0.30, "Thyroid normal in size and shape; heterogeneous right lobe with a hypoechoic nodule ~0.6x0.5 cm, well-defined and regular. CDFI: peripheral vascularity."),
        (0.18, "Both thyroid lobes enlarged with coarse heterogeneous echogenicity and a reticular pattern. CDFI: rich intraglandular vascularity."),
        (0.10, "Mixed-echo nodule ~1.4x1.1 cm in the left thyroid lobe with punctate echogenic foci and ill-defined margins; further evaluation advised."),
    ],
    "Hepatobiliary/spleen/pancreas/urinary/prostate ultrasound": [
        (0.40, "Liver normal in size and shape with smooth capsule and homogeneous echogenicity; intrahepatic vessels and bile ducts well visualized. Gallbladder normal in size and shape with smooth wall and no abnormal echogenicity. Spleen and pancreas unremarkable. Both kidneys normal in size and shape with no collecting-system separation. Prostate normal in size and shape with homogeneous echogenicity."),
        (0.28, "Liver normal in size and shape with increased and enhanced parenchymal echogenicity and deep attenuation, consistent with mild fatty liver. Gallbladder wall smooth with no abnormal echogenicity. Spleen and pancreas unremarkable. Both kidneys unremarkable. Prostate unremarkable."),
        (0.18, "Increased and enhanced hepatic parenchymal echogenicity consistent with moderate fatty liver. A ~4x3 mm echogenic projection without acoustic shadowing at the gallbladder wall, suggesting a gallbladder polyp. Both kidneys unremarkable. Punctate echogenic foci in the prostate, suggesting calcification."),
        (0.14, "A ~1.0x0.8 cm slightly echogenic mass in the liver, well-defined and homogeneous, possibly a hemangioma. Gallbladder, spleen and pancreas unremarkable. A ~0.4 cm echogenic focus in the right kidney, suggesting a stone."),
    ],
    "Liver/gallbladder/spleen/pancreas/urinary ultrasound": [
        (0.55, "Liver normal in size and shape with intact smooth capsule and homogeneous parenchymal echogenicity; intrahepatic vessels and bile ducts well visualized with normal course. Gallbladder normal in size and shape with smooth wall. Spleen and pancreas unremarkable. Both kidneys unremarkable."),
        (0.30, "Increased and enhanced hepatic parenchymal echogenicity consistent with mild fatty liver. Gallbladder wall smooth. Spleen and pancreas unremarkable. Both kidneys unremarkable."),
        (0.15, "Heterogeneous hepatic parenchymal echogenicity with scattered slightly echogenic areas, suggesting inhomogeneous fatty liver. Gallbladder wall coarse. Both kidneys unremarkable."),
    ],
    "3* Liver/gallbladder/pancreas/spleen ultrasound": [
        (0.6, "Liver normal in size and shape with homogeneous echogenicity. Gallbladder, pancreas and spleen unremarkable."),
        (0.4, "Increased and enhanced hepatic parenchymal echogenicity consistent with mild fatty liver. Gallbladder wall coarse; otherwise unremarkable."),
    ],
    "Echocardiography": [
        (0.50, "Cardiac chambers not enlarged; aortic and main pulmonary artery diameters normal. Atrial and ventricular septa intact; interventricular septum and left ventricular posterior wall not thickened with fair motion. Valve thickness, elasticity and opening normal. CDFI: no abnormal flow signals. Left ventricular ejection fraction 65%."),
        (0.28, "Cardiac chambers normal in size. Mild aortic regurgitation; trivial mitral and tricuspid regurgitation. Reduced left ventricular diastolic function; ejection fraction 62%."),
        (0.14, "Mild left atrial enlargement; interventricular septum thickened to ~12 mm; aortic valve calcification with mild regurgitation. Reduced left ventricular diastolic function; ejection fraction 60%."),
        (0.08, "Cardiac chambers normal in size with coordinated wall motion and no segmental abnormalities; ejection fraction 68%."),
    ],
    "Transcranial Doppler (TCD)": [
        (0.55, "Normal flow velocities and spectral waveforms in bilateral middle, anterior and posterior cerebral arteries; no obvious abnormality."),
        (0.30, "Increased flow velocity in bilateral middle cerebral arteries with acceptable spectral waveform, possibly vasospasm."),
        (0.15, "Decreased flow velocity in bilateral vertebrobasilar arteries, suggesting possible insufficiency."),
    ],
    "Carotid ultrasound": [
        (0.45, "Normal caliber of bilateral common, internal and external carotid arteries with smooth intima and no obvious plaque. CDFI: good luminal filling."),
        (0.32, "Thickened, echogenic and irregular bilateral carotid intima; a hypoechoic plaque ~0.56x0.15 cm at the posterior wall of the right common carotid bulb; no obvious luminal stenosis."),
        (0.15, "Multiple echogenic and mixed-echo plaques in bilateral carotid arteries, thickest ~0.22 cm, with ~30% luminal stenosis."),
        (0.08, "Thickened bilateral carotid intima-media without obvious plaque formation."),
    ],
    "Bilateral carotid-vertebral artery ultrasound": [
        (0.6, "Symmetric diameters of bilateral common, internal and external carotid and vertebral arteries; intima-media not thickened; normal flow velocities and spectral waveforms in all segments."),
        (0.4, "Thickened bilateral carotid intima-media with punctate echogenic plaques; normal flow velocities and spectral waveforms in all segments."),
    ],
    "3* Vertebral artery ultrasound": [
        (0.7, "Normal caliber of bilateral vertebral arteries with good luminal sonolucency, color flow filling and normal flow velocity."),
        (0.3, "Smaller caliber of the right vertebral artery with reduced flow velocity, possibly a developmental variant."),
    ],
    "Chest CT plain scan": [
        (0.40, "Clear lung markings with natural course and good lucency; no definite parenchymal lesion. Tracheobronchial tree patent. Mediastinum centered with no enlarged lymph nodes. Cardiac silhouette normal in size and shape. No bilateral pleural effusion."),
        (0.30, "Increased lung markings; a ~4 mm tiny nodule in the right upper lobe, well-defined; annual follow-up advised. No enlarged mediastinal lymph nodes. Cardiac silhouette not enlarged."),
        (0.18, "Scattered linear high-density opacities in both lungs with slightly blurred margins, suggesting chronic inflammatory changes. A few localized areas of emphysema in both lungs. No enlarged mediastinal lymph nodes."),
        (0.12, "A ~8 mm ground-glass nodule in the left lower lobe with ill-defined margins; re-examination in 3 months advised. No enlarged mediastinal lymph nodes."),
    ],
    "Chest CT plain scan (low dose)": [
        (0.6, "Clear lung markings with good lucency; no definite nodule or consolidation. No enlarged mediastinal lymph nodes."),
        (0.4, "A few linear high-density opacities in both lungs; a ~3 mm tiny nodule in the right middle lobe; follow-up advised."),
    ],
    "Chest CT contrast scan": [
        (0.6, "No definite space-occupying lesion in both lungs; no abnormal enhancement after contrast. No enlarged mediastinal lymph nodes."),
        (0.4, "A ~1.2 cm nodule in the right upper lobe with mild enhancement after contrast; further evaluation advised."),
    ],
    "Chest PA": [
        (0.55, "Symmetric thorax with no bony abnormality. Normal bilateral lung lucency without abnormal increased density. Slightly increased and prominent lung markings. No enlargement or increased density of hila. Cardiac silhouette normal in size and shape. Mediastinum centered; smooth bilateral diaphragms; sharp costophrenic angles."),
        (0.28, "Symmetric thorax. Increased and disordered lung markings. Hila not enlarged. Cardiac silhouette not enlarged. Smooth bilateral diaphragms; sharp costophrenic angles."),
        (0.17, "Aortic knob prominent to the left with widened aorta. Slightly increased lung markings. Slightly enlarged cardiac silhouette. Sharp bilateral costophrenic angles."),
    ],
    "Chest PA (film)": [
        (0.6, "Symmetric thorax with no bony abnormality. Normal bilateral lung lucency without abnormal increased density. Slightly increased lung markings. Cardiac silhouette normal in size and shape. Mediastinum centered; smooth bilateral diaphragms; sharp costophrenic angles."),
        (0.4, "Increased and disordered lung markings. Cardiac silhouette not enlarged. Smooth bilateral diaphragms; sharp costophrenic angles."),
    ],
    "Chest PA and lateral": [
        (0.5, "Symmetric thorax with mild thoracic vertebral osteophytosis. Normal bilateral lung lucency without abnormal increased density. Clear lung markings. Hila not enlarged. Cardiac silhouette normal in size and shape. Mediastinum centered; smooth bilateral diaphragms; sharp costophrenic angles."),
        (0.3, "Increased and disordered lung markings without definite consolidation. Cardiac silhouette not enlarged. Mild thoracic vertebral osteophytosis."),
        (0.2, "Aortic dilatation with the aortic knob prominent to the left. Slightly increased and blurred lung markings. Cardiac silhouette normal in size and shape."),
    ],
    "* Chest PA and lateral *": [
        (0.6, "Symmetric thorax. Normal bilateral lung lucency without definite consolidation. Slightly increased lung markings. Smooth bilateral diaphragms; clear sharp costophrenic angles. Mediastinum centered; cardiac silhouette normal in size and shape."),
        (0.4, "Aortic knob prominent to the left. Increased and blurred lung markings. Cardiac silhouette not enlarged. Sharp bilateral costophrenic angles."),
    ],
    "Chest PA and lateral (external)": [
        (0.7, "Symmetric thorax with normal bilateral lung lucency and no abnormal increased density. Cardiac silhouette normal in size and shape. Smooth bilateral diaphragms; sharp costophrenic angles."),
        (0.3, "Increased lung markings. Cardiac silhouette not enlarged. Sharp bilateral costophrenic angles."),
    ],
    "Chest PA (external)": [
        (0.7, "Symmetric thorax with normal bilateral lung lucency and no abnormal increased density. Cardiac silhouette normal in size and shape."),
        (0.3, "Increased and disordered lung markings. Cardiac silhouette not enlarged."),
    ],
    "Chest PA and lateral (external)": [
        (0.7, "Symmetric thorax with normal bilateral lung lucency and no abnormal increased density. Cardiac silhouette normal in size and shape."),
        (0.3, "Increased lung markings. Cardiac silhouette not enlarged."),
    ],
    "Chest CT plain scan (choose one)": [
        (0.7, "Clear lung markings without definite parenchymal lesion. No enlarged mediastinal lymph nodes. Cardiac silhouette not enlarged."),
        (0.3, "A few linear high-density opacities in both lungs, suggesting chronic inflammation."),
    ],
    "Bone density scan": [
        (0.42, "Normal bone mass; T-score -0.8, Z-score -0.5."),
        (0.33, "Osteopenia; T-score -1.6, Z-score -1.1; increased calcium and vitamin D intake and appropriate weight-bearing exercise advised."),
        (0.18, "Osteoporosis; T-score -2.8, Z-score -1.9; orthopedic evaluation for anti-osteoporosis therapy advised."),
        (0.07, "Normal bone mass; T-score -0.3, Z-score 0.2."),
    ],
    "liver_stiffness_noninvasive": [
        (0.55, "Liver stiffness 4.8 kPa; fat attenuation parameter 218 dB/m; no obvious liver fibrosis."),
        (0.28, "Liver stiffness 6.9 kPa; fat attenuation parameter 268 dB/m; suggesting mild liver fibrosis and moderate fatty liver."),
        (0.17, "Liver stiffness 9.6 kPa; fat attenuation parameter 312 dB/m; suggesting significant liver fibrosis; hepatology follow-up advised."),
    ],
    "Uterus and adnexa ultrasound": [
        (0.5, "Anteverted uterus normal in size and shape with smooth capsule and homogeneous echogenicity; endometrial line centered and clear. No abnormal cervical mass. No obvious abnormal echogenicity in bilateral adnexa."),
        (0.3, "Heterogeneous myometrial echogenicity with scattered punctate echogenic foci, possibly adenomyosis. Bilateral adnexa unremarkable."),
        (0.2, "A ~2.1x1.8 cm hypoechoic mass in the anterior uterine wall, well-defined, suggesting a uterine fibroid. Bilateral adnexa unremarkable."),
    ],
    "Transvaginal ultrasound": [
        (0.55, "Anteverted uterus normal in size and shape with homogeneous echogenicity; endometrial cavity line centered. Bilateral ovaries normal in size with no abnormal echogenicity."),
        (0.3, "A ~1.5x1.2 cm hypoechoic mass in the posterior uterine wall, well-defined, suggesting a uterine fibroid. Bilateral adnexa unremarkable."),
        (0.15, "Endometrium thickened to ~1.2 cm with heterogeneous echogenicity; gynecology follow-up advised."),
    ],
    "Urinary ultrasound (female)": [
        (0.6, "Both kidneys normal in size and shape with no collecting-system separation. No bilateral ureteral dilatation. Bladder well filled with smooth wall."),
        (0.4, "A ~0.4 cm echogenic focus with posterior acoustic shadowing in the right kidney, suggesting a stone. Bladder unremarkable."),
    ],
    "Urinary ultrasound (male)": [
        (0.55, "Both kidneys normal in size and shape with no collecting-system separation. Bladder well filled. Prostate normal in size and shape with homogeneous echogenicity."),
        (0.45, "Enlarged prostate with heterogeneous echogenicity and punctate echogenic foci, suggesting benign prostatic hyperplasia with calcification. Both kidneys unremarkable."),
    ],
    "Breast ultrasound": [
        (0.45, "Homogeneous echogenicity of bilateral breast glands; no obvious space-occupying lesion. CDFI: no abnormal color flow signals. No obviously abnormal axillary lymph nodes."),
        (0.30, "Somewhat heterogeneous echogenicity of bilateral breast glands with scattered patchy hypoechoic areas; no obvious mass. No abnormal axillary lymph nodes."),
        (0.17, "A ~0.8x0.6 cm hypoechoic nodule in the upper outer quadrant of the right breast, well-defined and regular, BI-RADS 3; follow-up advised."),
        (0.08, "A ~1.1x0.9 cm hypoechoic nodule in the left breast, ill-defined and irregular, BI-RADS 4a; further workup advised."),
    ],
    "* Bilateral breast ultrasound *": [
        (0.6, "Enhanced cord-like echogenicity of bilateral breast glands with somewhat heterogeneous distribution and a few scattered patchy hypoechoic areas. CDFI: no abnormal color flow."),
        (0.4, "Somewhat heterogeneous echogenicity of bilateral breast glands; a ~0.7 cm hypoechoic nodule in the right breast, BI-RADS 3."),
    ],
    "* Bilateral mammography *": [
        (0.6, "Roundish dense shadows of bilateral breast glands without definite mass or malignant calcification. BI-RADS 2."),
        (0.4, "Clustered fine punctate calcifications in the right breast, BI-RADS 3; follow-up advised."),
    ],
    "Liquid-based thin-layer cytology (photomicrography)": [
        (0.6, "Specimen adequacy: satisfactory. Total cell count: >5000. Pathogens: no Trichomonas, fungal, herpesvirus-infected or HPV-infected cells. Cell findings: squamous epithelial cells present; no intraepithelial lesion cells. Diagnosis: negative for intraepithelial lesion or malignancy (NILM)."),
        (0.25, "Specimen adequacy: satisfactory. Atypical squamous cells of undetermined significance (ASC-US); HPV testing and follow-up advised."),
        (0.15, "Specimen adequacy: satisfactory. Low-grade squamous intraepithelial lesion (LSIL); colposcopy for further evaluation advised."),
    ],
    "Liquid-based cytology + cervical cell computer diagnosis": [
        (0.65, "Specimen adequacy: satisfactory. Negative for intraepithelial lesion or malignancy (NILM)."),
        (0.35, "Specimen adequacy: satisfactory. Atypical squamous cells of undetermined significance (ASC-US); follow-up advised."),
    ],
    "Cervical exfoliative cytology and diagnosis (photomicrography)": [
        (0.7, "Negative for intraepithelial lesion or malignancy (NILM)."),
        (0.3, "Atypical squamous cells of undetermined significance (ASC-US); HPV testing advised."),
    ],
    "Esophagogastroduodenoscopy (Gastroenterology)": [
        (0.35, "Endoscope inserted smoothly to the upper descending duodenum. Esophageal mucosa smooth with good peristalsis. Cardia opens and closes well; E-G line clear. Gastric fundus mucus pool clear with smooth mucosa. Gastric body mucosa smooth with normal color. Gastric angle arc-shaped and smooth. Antral mucosa smooth with good peristalsis. Pylorus round and opens/closes well. Duodenal bulb and descending part unremarkable. Diagnosis: chronic non-atrophic gastritis."),
        (0.30, "Mottled hyperemia of the antral mucosa with scattered linear erosions; no varices. Duodenum unremarkable. Diagnosis: chronic atrophic gastritis with erosion."),
        (0.20, "A ~0.6 cm ulcer on the lesser curvature of the lower gastric body covered with white exudate and surrounding mucosal hyperemia and edema. Diagnosis: gastric ulcer (active); biopsy advised."),
        (0.15, "A ~0.5 cm mucosal break in the lower esophagus. Diagnosis: reflux esophagitis (LA grade A)."),
    ],
    "Esophagogastroduodenoscopy (Area B)": [
        (0.5, "Esophageal mucosa smooth. Gastric fundus mucus pool slightly turbid with smooth mucosa. Gastric body mucosa smooth. Mild antral mucosal hyperemia. Duodenum unremarkable. Diagnosis: chronic non-atrophic gastritis."),
        (0.5, "Mottled hyperemia of the antral mucosa with scattered erosions. Diagnosis: chronic atrophic gastritis with erosion."),
    ],
    "Esophagogastroduodenoscopy (VIP)": [
        (0.6, "Esophagus, stomach and duodenum unremarkable."),
        (0.4, "Chronic non-atrophic gastritis."),
    ],
    "Esophagogastroduodenoscopy (choose one)": [
        (0.6, "Chronic non-atrophic gastritis."),
        (0.4, "Chronic atrophic gastritis with erosion."),
    ],
    "Colonoscopy (Gastroenterology)": [
        (0.40, "Bowel preparation good. Routine colonoscopy to the cecum. Entire colonic mucosa smooth with clear vascular pattern; no obvious abnormality. Diagnosis: colon unremarkable."),
        (0.30, "A flat polyp ~0.5 cm in diameter at the sigmoid colon 30 cm from the anal verge, with smooth surface, removed by biopsy forceps. Diagnosis: sigmoid colon polyp (removed)."),
        (0.20, "A ~0.8 cm semi-pedunculated polyp at the ileocecal region with surface hyperemia, removed by snare. Diagnosis: ileocecal polyp (removed)."),
        (0.10, "Rectal mucosal hyperemia and edema with blurred vascular pattern. Diagnosis: proctitis."),
    ],
    "Colonoscopy (Area B)": [
        (0.6, "Routine colonoscopy to the terminal ileum; terminal ileal mucosa unremarkable. Entire colonic mucosa smooth with clear vascular pattern; no obvious abnormality."),
        (0.4, "A ~0.4 cm polyp in the sigmoid colon, removed by forceps."),
    ],
    "Colonoscopy (VIP)": [
        (0.7, "Entire colonic mucosa smooth; no obvious abnormality."),
        (0.3, "A ~0.5 cm polyp in the transverse colon, removed by forceps."),
    ],
    "Colonoscopy": [
        (0.7, "Entire colonic mucosa smooth; no obvious abnormality."),
        (0.3, "Sigmoid colon polyp (removed)."),
    ],
    "Cervical spine AP/lateral": [
        (0.45, "Cervical lordosis slightly straightened with continuous alignment. Mild lip-like osteophytosis at the margins of C4-C7. Remaining vertebrae and appendages normal in shape and size; no abnormal disc-space narrowing."),
        (0.30, "Straightened cervical lordosis with vertebral marginal osteophytosis and slightly narrowed disc spaces."),
        (0.25, "Normal cervical lordosis with continuous alignment; no obvious bony abnormality."),
    ],
    "* Cervical spine AP/lateral *": [
        (0.6, "Cervical lordosis slightly straightened with mild vertebral marginal osteophytosis; no obvious disc-space narrowing."),
        (0.4, "Normal cervical lordosis; no obvious bony abnormality."),
    ],
    "* Cervical spine AP/lateral * (choose one)": [
        (0.6, "Cervical lordosis slightly straightened with mild vertebral marginal osteophytosis."),
        (0.4, "No obvious cervical bony abnormality."),
    ],
    "Cervical spine AP/lateral (external)": [
        (0.7, "Normal cervical lordosis with continuous alignment; no obvious bony abnormality."),
        (0.3, "Mild osteophytosis at the cervical vertebral margins."),
    ],
    "Lumbar spine AP/lateral": [
        (0.45, "Lumbar lordosis preserved with continuous alignment. Lip-like osteophytosis at the margins of L3-L5. No obvious disc-space narrowing."),
        (0.30, "Straightened lumbar lordosis with vertebral marginal osteophytosis and slightly narrowed L4-L5 disc space."),
        (0.25, "No obvious lumbar bony abnormality."),
    ],
    "* Lumbar spine AP/lateral *": [
        (0.6, "Mild osteophytosis at the lumbar vertebral margins; no obvious disc-space narrowing."),
        (0.4, "No obvious lumbar bony abnormality."),
    ],
    "Thoracic spine AP/lateral": [
        (0.6, "Thoracic kyphosis preserved with mild vertebral marginal osteophytosis; no obvious disc-space narrowing."),
        (0.4, "No obvious thoracic bony abnormality."),
    ],
    "Head CT plain scan": [
        (0.45, "A few patchy low-density areas beside the bilateral lateral ventricles and centrum semiovale with partially clear margins. Mild enlargement of cisterns and widening of sulci; midline structures centered. Diagnosis: mild white-matter ischemic changes; brain atrophy."),
        (0.30, "No definite abnormal density in the brain parenchyma; ventricular system normal in size and shape; midline structures centered."),
        (0.25, "Lacunar low-density foci in bilateral basal ganglia, suggesting lacunar infarction. Brain atrophy."),
    ],
    "MRI brain plain scan (1.5T)": [
        (0.5, "Scattered punctate long-T1/long-T2 signals beside the bilateral centrum semiovale and lateral ventricles, hyperintense on FLAIR. Ventricular system not enlarged; midline centered. Diagnosis: white-matter ischemic changes."),
        (0.5, "No definite abnormal signal in the brain parenchyma; ventricular system normal in size and shape; midline structures centered."),
    ],
    "MRI brain plain scan (3T)": [
        (0.5, "Scattered punctate ischemic foci in the bilateral centrum semiovale. Ventricular system not enlarged; midline centered."),
        (0.5, "No definite abnormal signal in the brain parenchyma."),
    ],
    "MRI brain (plain) MRA (3T)": [
        (0.5, "Intracranial arteries follow a natural course without definite stenosis or aneurysm. No abnormal parenchymal signal."),
        (0.5, "Mild luminal stenosis of the M1 segment of bilateral middle cerebral arteries. A few white-matter ischemic foci."),
    ],
    "MRI brain contrast scan (3T)": [
        (0.7, "No abnormal enhancement in the brain parenchyma."),
        (0.3, "A ~0.8 cm ring-enhancing lesion in the right frontal lobe; further evaluation advised."),
    ],
    "MRI brain (contrast) MRA (3T)": [
        (0.7, "No definite abnormality of intracranial arteries."),
        (0.3, "Focal stenosis of the C4 segment of the left internal carotid artery."),
    ],
    "Thoracic spine MRI (3T)": [
        (0.6, "Normal thoracic curvature; uniform vertebral signal; no obvious disc herniation."),
        (0.4, "Mild T6-7 disc herniation with mild dural sac compression."),
    ],
    "Lumbosacral spine MRI (1.5T)": [
        (0.5, "Straightened lumbar curvature; L4-5 and L5-S1 disc bulging with mild dural sac compression."),
        (0.5, "No obvious lumbar disc herniation; uniform vertebral signal."),
    ],
    "Cervical spine MRI (1.5T)": [
        (0.5, "Straightened cervical curvature; C4-5 and C5-6 disc herniation with dural sac compression."),
        (0.5, "No obvious cervical disc herniation."),
    ],
    "Cervical spine MRI (3T)": [
        (0.5, "Straightened cervical curvature; C5-6 disc herniation."),
        (0.5, "No obvious cervical disc herniation."),
    ],
    "Cervical CT plain scan (+3D bone)": [
        (0.6, "Osteophytosis at cervical vertebral margins; no obvious disc-space narrowing."),
        (0.4, "No obvious cervical bony abnormality."),
    ],
    "Lumbar CT plain scan (3 vertebrae)": [
        (0.6, "Osteophytosis at lumbar vertebral margins; L4-5 disc bulging."),
        (0.4, "No obvious lumbar abnormality."),
    ],
    "Bilateral kidney CT contrast scan": [
        (0.6, "Both kidneys normal in size and shape; no abnormal enhancement."),
        (0.4, "~0.5 cm non-enhancing low-density lesion in the right kidney, likely a cyst."),
    ],
    "Upper-abdomen CT contrast scan (liver/gallbladder/pancreas/spleen)": [
        (0.6, "No abnormal hepatic enhancement; gallbladder, pancreas and spleen unremarkable."),
        (0.4, "~1.2 cm mildly enhancing lesion in the right hepatic lobe, likely a hemangioma."),
    ],
    "Coronary CT angiography (3D)": [
        (0.5, "Coronary arteries follow a normal course; no obvious stenosis or calcification."),
        (0.5, "Punctate calcification in the proximal LAD with mild stenosis."),
    ],
    "Abdominal plain film (supine)": [
        (0.7, "Bowel gas present; no definite air-fluid levels or free air."),
        (0.3, "No definite abdominal abnormality."),
    ],
    "* Abdominal plain film (supine) *": [
        (0.7, "Bowel gas present; no definite abnormality."),
        (0.3, "No definite abdominal abnormality."),
    ],
    "3* Bilateral lower-limb artery ultrasound": [
        (0.6, "Lower-limb arterial intima-media not thickened; good luminal filling; no obvious stenosis."),
        (0.4, "Lower-limb arterial intima-media thickened with punctate echogenic plaque; fair luminal filling."),
    ],
    "3* Bilateral lower-limb vein ultrasound": [
        (0.7, "Lower-limb veins of normal caliber; no abnormal echogenicity; good filling; no thrombus."),
        (0.3, "Slow flow in the left peroneal vein; no definite thrombus."),
    ],
    "3* Salivary glands and cervical lymph nodes superficial ultrasound": [
        (0.7, "Both parotid and submandibular glands normal in size/shape with uniform echogenicity; no obviously enlarged cervical lymph nodes."),
        (0.3, "Several cervical lymph nodes, largest ~0.8x0.4 cm, with clear corticomedullary differentiation."),
    ],
    "3* Superficial mass ultrasound": [
        (0.6, "~0.9x0.5 cm well-defined hypoechoic area in the subcutaneous tissue at the mass site, possibly a lipoma."),
        (0.4, "No definite abnormal echogenicity at the mass site."),
    ],
    "Flexible laryngoscopy (electronic)": [
        (0.6, "Smooth nasopharyngeal mucosa with slight lymphoid follicle hyperplasia on the posterior pharyngeal wall; smooth laryngeal mucosa; both vocal cords mobile with good closure."),
        (0.4, "Chronic pharyngitis; no obvious abnormality of either vocal cord."),
    ],
    "Fundus photography": [
        (0.5, "Both optic discs well defined, pinkish, C/D ~0.3; retinal vessels normal; no hemorrhage or exudate."),
        (0.5, "Retinal arteries narrowed with increased light reflex and scattered dot hemorrhages, suggesting hypertensive retinopathy."),
    ],
    "* Right knee AP/lateral *": [
        (0.6, "Right knee joint space normal; no obvious bony abnormality."),
        (0.4, "Right knee osteophytosis with slightly narrowed joint space."),
    ],
    "* Left knee AP/lateral *": [
        (0.6, "Left knee joint space normal; no obvious bony abnormality."),
        (0.4, "Left knee osteophytosis with slightly narrowed joint space."),
    ],
    "* Unilateral shoulder AP/lateral *": [
        (0.6, "Shoulder joint space normal; no obvious bony abnormality."),
        (0.4, "Osteophytosis around the shoulder joint."),
    ],
    "Right ankle AP/lateral": [
        (0.7, "Right ankle joint space normal; no obvious bony abnormality."),
        (0.3, "Right ankle osteophytosis."),
    ],
    "Left wrist AP/lateral": [
        (0.7, "Left wrist joint space normal; no obvious bony abnormality."),
        (0.3, "Left wrist osteophytosis."),
    ],
}

# Exam items not explicitly listed in the template library use this fallback template
DESC_FALLBACK = [
    (0.7, "No obvious abnormality on this examination."),
    (0.3, "Mild abnormality on this examination; correlate clinically and follow up with the specialty."),
]

# Exam item -> ordering department (for PES_apply_item.det_dep_name)
DEPT_MAP = {
    "ecg": "ECG Room", "Holter ECG": "ECG Room", "Ambulatory BP monitoring": "ECG Room",
    "Thyroid ultrasound": "Ultrasound", "Hepatobiliary/spleen/pancreas/urinary/prostate ultrasound": "Ultrasound",
    "Liver/gallbladder/spleen/pancreas/urinary ultrasound": "Ultrasound", "3* Liver/gallbladder/pancreas/spleen ultrasound": "Ultrasound",
    "Echocardiography": "Ultrasound", "Carotid ultrasound": "Ultrasound", "Bilateral carotid-vertebral artery ultrasound": "Ultrasound",
    "3* Vertebral artery ultrasound": "Ultrasound", "Uterus and adnexa ultrasound": "Ultrasound", "Transvaginal ultrasound": "Ultrasound",
    "Breast ultrasound": "Ultrasound", "* Bilateral breast ultrasound *": "Ultrasound", "Urinary ultrasound (female)": "Ultrasound",
    "Urinary ultrasound (male)": "Ultrasound", "3* Bilateral lower-limb artery ultrasound": "Ultrasound",
    "3* Bilateral lower-limb vein ultrasound": "Ultrasound", "3* Salivary glands and cervical lymph nodes superficial ultrasound": "Ultrasound",
    "3* Superficial mass ultrasound": "Ultrasound", "Transcranial Doppler (TCD)": "Neurology",
    "Bone density scan": "Orthopedics", "liver_stiffness_noninvasive": "Hepatology",
    "Chest CT plain scan": "Radiology", "Chest CT plain scan (low dose)": "Radiology", "Chest CT contrast scan": "Radiology",
    "Chest CT plain scan (choose one)": "Radiology", "Head CT plain scan": "Radiology", "Cervical CT plain scan (+3D bone)": "Radiology",
    "Lumbar CT plain scan (3 vertebrae)": "Radiology", "Bilateral kidney CT contrast scan": "Radiology",
    "Upper-abdomen CT contrast scan (liver/gallbladder/pancreas/spleen)": "Radiology", "Coronary CT angiography (3D)": "Radiology",
    "MRI brain plain scan (1.5T)": "Radiology", "MRI brain plain scan (3T)": "Radiology",
    "MRI brain (plain) MRA (3T)": "Radiology", "MRI brain contrast scan (3T)": "Radiology",
    "MRI brain (contrast) MRA (3T)": "Radiology", "Thoracic spine MRI (3T)": "Radiology",
    "Lumbosacral spine MRI (1.5T)": "Radiology", "Cervical spine MRI (1.5T)": "Radiology",
    "Cervical spine MRI (3T)": "Radiology",
    "Liquid-based thin-layer cytology (photomicrography)": "Gynecology", "Liquid-based cytology + cervical cell computer diagnosis": "Gynecology",
    "Cervical exfoliative cytology and diagnosis (photomicrography)": "Gynecology",
    "Esophagogastroduodenoscopy (Gastroenterology)": "Gastroenterology", "Esophagogastroduodenoscopy (Area B)": "Gastroenterology",
    "Esophagogastroduodenoscopy (VIP)": "Gastroenterology", "Esophagogastroduodenoscopy (choose one)": "Gastroenterology",
    "Colonoscopy (Gastroenterology)": "Gastroenterology", "Colonoscopy (Area B)": "Gastroenterology",
    "Colonoscopy (VIP)": "Gastroenterology", "Colonoscopy": "Gastroenterology",
    "Flexible laryngoscopy (electronic)": "ENT", "Fundus photography": "Ophthalmology",
    "* Bilateral mammography *": "Radiology",
}


# ---------------------------------------------------------------- Phenotype injection
# Purpose: purely independent sampling makes indicators uncorrelated, pushing all risk models to low risk,
# No patient qualifies at the report stage. Inject clinically coherent phenotypes for some patients (abnormal indicators + descriptions),
# Give demo data realistic disease-clustering characteristics.
#
# lab  : item name -> (lower, upper); sample uniformly within the interval
# desc : [(exam item name, variant index in DESC_TEMPLATES)] forces that exam to appear with the given variant

PHENOTYPES = {
    "diabetes": {
        "lab": {
            "Glucose (GLU)": (7.2, 12.5),
            "Glycated hemoglobin (GHb)": (6.8, 9.8),
        },
        "desc": [],
    },
    "hypertension": {
        "lab": {
            "Right Upper Arm Systolic Blood Pressure (RbSYS)": (145, 178),
            "Right Upper Arm Diastolic Blood Pressure (RbDIA)": (92, 108),
            "Left Upper Arm Systolic Blood Pressure (LbSYS)": (143, 176),
            "Left Upper Arm Diastolic Blood Pressure (LbDIA)": (90, 106),
        },
        "desc": [],
    },
    "dyslipidemia": {
        "lab": {
            "TG": (2.4, 5.2),
            "CHOL": (5.9, 7.8),
            "Low density lipoprotein cholesterol (LDL)": (3.7, 5.2),
            "High density lipoprotein cholesterol (HDL)": (0.62, 0.88),
        },
        "desc": [],
    },
    "hyperuricemia": {
        "lab": {"Uric Acid (UA)": (465, 640)},
        "desc": [],
    },
    "renal_impairment": {
        "lab": {"Creatinine (Cr)": (112, 168)},
        "desc": [],
    },
    "central_obesity": {
        "lab": {
            "Waist Circumference": (96, 112),
            "Weight": (82, 98),
        },
        "desc": [],
    },
    "fatty_liver": {
        "lab": {
            "TG": (2.1, 4.2),
            "Uric Acid (UA)": (420, 560),
        },
        "desc": [
            ("Hepatobiliary/spleen/pancreas/urinary/prostate ultrasound", 1),
            ("Liver/gallbladder/spleen/pancreas/urinary ultrasound", 1),
        ],
    },
    "thyroid_nodule": {
        "lab": {},
        "desc": [("Thyroid ultrasound", 1)],
    },
    "osteoporosis": {
        "lab": {},
        "desc": [("Bone density scan", 2)],
    },
    "carotid_plaque": {
        "lab": {
            "CHOL": (5.6, 7.2),
            "Low density lipoprotein cholesterol (LDL)": (3.5, 4.8),
        },
        "desc": [("Carotid ultrasound", 1)],
    },
    "emphysema": {
        "lab": {},
        "desc": [("Chest CT plain scan", 2), ("Chest PA", 1)],
    },
    "colorectal_polyp": {
        "lab": {},
        "desc": [("Colonoscopy (Gastroenterology)", 1)],
    },
    "breast_nodule": {
        "lab": {},
        "desc": [("Breast ultrasound", 2)],
    },
    "benign_prostatic_hyperplasia": {
        "lab": {},
        "desc": [("Urinary ultrasound (male)", 1)],
    },
    "chronic_gastritis": {
        "lab": {},
        "desc": [("Esophagogastroduodenoscopy (Gastroenterology)", 1)],
    },
}

# Phenotype sampling weights (earlier entries are more common)
PHENOTYPE_WEIGHTS = {
    "fatty_liver": 14, "thyroid_nodule": 13, "carotid_plaque": 11, "dyslipidemia": 10,
    "hypertension": 9, "diabetes": 8, "hyperuricemia": 7, "osteoporosis": 6,
    "central_obesity": 6, "chronic_gastritis": 5, "benign_prostatic_hyperplasia": 4, "emphysema": 4,
    "colorectal_polyp": 3, "breast_nodule": 3, "renal_impairment": 2,
}


# ---------------------------------------------------------------- Utility functions

def _is_num(v):
    try:
        float(str(v).strip())
        return True
    except Exception:
        return False


def learn_lab_profile():
    """Learn each lab item's distribution from local historical batches (aggregate statistics only, no patient rows)."""
    files = sorted(glob.glob(BATCH_GLOB))
    if not files:
        sys.exit(f"[FAIL] no historical batch data found: {BATCH_GLOB}")

    buckets = defaultdict(list)
    meta = {}
    per_patient_counts = []
    item_patient_hits = Counter()
    total_patients = 0

    for f in files:
        try:
            df = pd.read_csv(f, dtype=str, low_memory=False)
        except Exception:
            continue
        if "res_itm_name" not in df.columns:
            continue
        df = df[df["res_itm_name"].notna()]
        df = df[df["res_itm_name"].astype(str).str.strip() != ""]
        if df.empty:
            continue

        if "app_no" in df.columns:
            per_patient_counts.append(df.groupby("app_no").size().values)
            total_patients += df["app_no"].nunique()
            for n in df["res_itm_name"].dropna().unique():
                item_patient_hits[str(n).strip()] += df.loc[
                    df["res_itm_name"] == n, "app_no"
                ].nunique()

        for name, grp in df.groupby("res_itm_name"):
            name = str(name).strip()
            vals = [str(v).strip() for v in grp["res_value"].tolist() if str(v).strip() not in ("", "nan")]
            if not vals:
                continue
            buckets[name].extend(vals)
            if name not in meta:
                row = grp.iloc[0]
                meta[name] = {
                    "res_itm_id": str(row.get("res_itm_id") or "").strip(),
                    "res_unit": str(row.get("res_unit") or "").strip(),
                    "res_exp": str(row.get("res_exp") or "").strip(),
                    "res_refl": str(row.get("res_refl") or "").strip(),
                    "res_refu": str(row.get("res_refu") or "").strip(),
                    "res_specimen": str(row.get("res_specimen") or "").strip(),
                }

    profile = {}
    for name, vals in buckets.items():
        numeric = [float(v) for v in vals if _is_num(v)]
        ratio = len(numeric) / len(vals) if vals else 0
        m = meta.get(name, {})
        if ratio >= 0.7 and len(numeric) >= 5:
            arr = np.array(numeric, dtype=float)
            q = np.percentile(arr, [5, 25, 50, 75, 95])
            profile[name] = {
                "kind": "num",
                "p05": float(q[0]), "p25": float(q[1]), "p50": float(q[2]),
                "p75": float(q[3]), "p95": float(q[4]),
                "decimals": _guess_decimals(arr),
                **m,
            }
        else:
            cnt = Counter(vals)
            top = cnt.most_common(6)
            total = sum(c for _, c in top)
            profile[name] = {
                "kind": "cat",
                "values": [v for v, _ in top],
                "weights": [c / total for _, c in top],
                **m,
            }

    counts = np.concatenate(per_patient_counts) if per_patient_counts else np.array([150])
    item_count_stats = {
        "p10": int(np.percentile(counts, 10)),
        "p50": int(np.percentile(counts, 50)),
        "p90": int(np.percentile(counts, 90)),
    }

    # Core item set: items present in >=70% of real patients, keeping synthetic patients complete
    core_items = [
        n for n, hits in item_patient_hits.items()
        if total_patients and hits / total_patients >= 0.70 and n in profile
    ]
    core_items.sort()
    return profile, item_count_stats, core_items


def _guess_decimals(arr):
    """Infer a sensible number of decimals from the magnitude."""
    med = float(np.median(np.abs(arr)))
    if med >= 1000:
        return 0
    if med >= 100:
        return 1
    if med >= 10:
        return 1
    return 2


def sample_lab_value(spec, rng):
    if spec["kind"] == "cat":
        return str(rng.choice(spec["values"], p=spec["weights"]))
    p25, p50, p75 = spec["p25"], spec["p50"], spec["p75"]
    sigma = max((p75 - p25) / 1.349, abs(p50) * 0.08, 1e-6)
    val = rng.normal(p50, sigma)
    lo = min(spec["p05"], p25) - 3 * sigma
    hi = max(spec["p95"], p75) + 3 * sigma
    val = float(np.clip(val, lo, hi))
    if val < 0 and spec["p05"] >= 0:
        val = abs(val) * 0.5
    d = spec["decimals"]
    return f"{val:.{d}f}" if d > 0 else str(int(round(val)))


def pick_desc(name, rng):
    tmpl = DESC_TEMPLATES.get(name) or DESC_FALLBACK
    texts = [t for _, t in tmpl]
    weights = [w for w, _ in tmpl]
    s = sum(weights)
    return str(rng.choice(texts, p=[w / s for w in weights]))


# ---------------------------------------------------------------- Main flow

def build(count=5000, seed=20261110):
    rng = np.random.default_rng(seed)
    print("1) Learning lab-item distributions (aggregate statistics only)...")
    profile, item_stats, core_items = learn_lab_profile()
    print(f"   lab items {len(profile)}; core items {len(core_items)}; "
          f"items per patient p10/p50/p90 = "
          f"{item_stats['p10']}/{item_stats['p50']}/{item_stats['p90']}")

    names = list(profile.keys())
    # Item frequency: approximate from real-data counts (uniform plus high-frequency weighting)
    freq = np.array([1.0] * len(names))
    for i, n in enumerate(names):
        if profile[n]["kind"] == "num":
            freq[i] = 1.0 + min(profile[n]["p50"] > 0, 1) * 0.0
    freq = freq / freq.sum()

    desc_names = list(DESC_TEMPLATES.keys())
    desc_freq = np.array([1.0] * len(desc_names))
    desc_freq = desc_freq / desc_freq.sum()

    phen_names = list(PHENOTYPE_WEIGHTS.keys())
    phen_w = np.array([PHENOTYPE_WEIGHTS[p] for p in phen_names], dtype=float)
    phen_w = phen_w / phen_w.sum()

    print(f"2) Generating {count} anonymized patients (with phenotype injection)...")
    apply_rows, patient_rows, result_rows, item_rows, desc_rows, summary_rows = [], [], [], [], [], []

    detail_id = DETAIL_ID_START
    base_date = pd.Timestamp("2024-01-01")

    for i in range(count):
        app_no = str(APP_NO_START + i)
        pat_id = f"{PAT_ID_PREFIX}{i + 1:06d}"
        name = f"Subject{i + 1:04d}"
        sex = "1" if i % 2 == 0 else "2"
        age = int(np.clip(rng.normal(52, 13), 20, 88))
        birth = (base_date
                 - pd.Timedelta(days=int(age * 365.25) + int(rng.integers(0, 365)))
                 ).strftime("%Y-%m-%d")
        app_time = (base_date
                    + pd.Timedelta(days=int(rng.integers(0, 700)))
                    + pd.Timedelta(hours=int(rng.integers(7, 11)))
                    ).strftime("%Y-%m-%d %H:%M:%S")
        check_date = app_time[:10]
        address = ADDRESS_TMPL.format(n=int(rng.integers(1, 999)))

        # --- PES_APPLY
        a = {c: "" for c in COLS_APPLY}
        a.update({
            "app_no": app_no, "pat_id": pat_id, "app_pat_name": name,
            "app_pat_age": str(age), "app_time": app_time,
            "app_check_date": check_date, "app_org_id": "DEMO0001",
            "app_flag": "50", "app_co_id": "90001", "app_co_batch": "1",
            "app_co_group": "1", "app_co_depart": "Health Examination Center",
            "app_signin_flag": "0", "app_signout_flag": "0", "app_seq": "0",
            "app_price": f"{float(rng.integers(800, 6000)):.2f}",
            "app_del_flag": "0", "app_pat_agemonth": "0",
            "app_rep_takemode": "Self pickup", "app_source": "1",
            "app_level": "1", "app_breakfast": "0", "org_id": "DEMO0001",
            "app_rep_mode": "0", "app_split_no": "0",
        })
        apply_rows.append(a)

        # --- PES_PATIENT (all PII columns left blank)
        p = {c: "" for c in COLS_PATIENT}
        p.update({
            "pat_id": pat_id, "pat_name": name, "pat_sex": sex,
            "pat_birth_date": birth, "pat_nationality": NATIONALITY,
            "pat_nation": NATION, "pat_address": address,
            "pat_marriage": "Married" if age >= 26 else "Single",
            "pat_job": "other", "pat_del_flag": "0", "pat_valid": "1",
            "pat_create_time": app_time, "pat_create_userid": "demo",
            "pat_create_username": "demo", "pat_lastupdate_time": app_time,
            "pat_level_id": "1", "pat_view_elecreport": "1",
            "pat_login_flag": "0", "pat_idcard_type": "1",
        })
        patient_rows.append(p)

        # --- Phenotype injection: about 45% of patients carry 1-2 phenotypes so risk models yield positives
        n_phen = 0
        if rng.random() < 0.45:
            n_phen = 1 if rng.random() < 0.7 else 2
        chosen = list(rng.choice(phen_names, size=n_phen, replace=False, p=phen_w)) if n_phen else []
        lab_override = {}
        desc_force = {}
        for pn in chosen:
            spec = PHENOTYPES[pn]
            for it, (lo, hi) in spec["lab"].items():
                lab_override[it] = float(rng.uniform(lo, hi))
            for exam, vi in spec["desc"]:
                desc_force[exam] = vi

        # --- Lab details: core item set + random supplements
        n_items = int(np.clip(rng.normal(item_stats["p50"], 20), 40, 260))
        chosen_items = list(core_items)
        extra_pool = [n for n in names if n not in set(core_items)]
        n_extra = max(0, min(n_items - len(chosen_items), len(extra_pool)))
        if n_extra:
            ex_freq = np.array([1.0] * len(extra_pool))
            ex_freq = ex_freq / ex_freq.sum()
            chosen_items += [extra_pool[int(j)] for j in
                             rng.choice(len(extra_pool), size=n_extra, replace=False, p=ex_freq)]
        # Add items involved in the phenotype but missing from the list
        for it in lab_override:
            if it in profile and it not in chosen_items:
                chosen_items.append(it)

        for j, it in enumerate(chosen_items):
            spec = profile[it]
            detail_id += 1
            did = str(detail_id)
            if it in lab_override:
                d = spec.get("decimals", 1)
                val = f"{lab_override[it]:.{d}f}" if d > 0 else str(int(round(lab_override[it])))
            else:
                val = sample_lab_value(spec, rng)
            result_rows.append({
                "detail_id": did, "res_itm_id": spec.get("res_itm_id", ""),
                "app_no": app_no, "res_com_id": "", "res_value": val,
                "res_value2": "", "res_unit": spec.get("res_unit", ""),
                "res_refl": spec.get("res_refl", ""), "res_refu": spec.get("res_refu", ""),
                "res_info": "", "res_date": app_time, "res_seq": str(j + 1),
                "res_exp": spec.get("res_exp", ""), "res_value3": "", "res_value4": "",
                "res_itm_name": it, "res_specimen": spec.get("res_specimen", ""),
                "res_sum_flag": "0", "res_value_num": val if _is_num(val) else "",
                "res_panic": "0",
            })
            item_rows.append({
                "detail_id": did, "det_time": app_time, "app_no": app_no,
                "pat_id": pat_id, "det_com_id": "", "det_com_name": it,
                "det_dep_id": "", "det_dep_name": "Laboratory",
                "det_price": "0.00", "det_amount": "1.0", "det_discount": "1.0",
                "det_totalprice": "0.00", "det_discount_totalprice": "0.00",
                "det_charge_flag": "1", "det_charge_type": "1", "det_rep_type": "0",
                "det_userid": "demo", "det_username": "demo", "det_usercode": "demo",
                "det_flag": "0", "det_signin_flag": "0", "det_append_flag": "0",
                "det_aba_flag": "0", "det_delay_flag": "0", "det_apply_sendflag": "0",
                "det_del_flag": "0", "det_balance_flag": "0", "det_restype": "0",
                "det_add_item": "0", "det_checkself": "0", "det_fee_type": "0",
                "det_extend_flag": "0",
            })

        # --- Descriptive exams (phenotype-mandated items first)
        n_desc = int(rng.integers(3, 9))
        forced = list(desc_force.keys())
        pool = [n for n in desc_names if n not in desc_force]
        n_rand = max(0, min(n_desc - len(forced), len(pool)))
        rand_pick = []
        if n_rand:
            pf = np.array([1.0] * len(pool))
            pf = pf / pf.sum()
            rand_pick = [pool[int(j)] for j in
                         rng.choice(len(pool), size=n_rand, replace=False, p=pf)]

        for dn in forced + rand_pick:
            detail_id += 1
            did = str(detail_id)
            if dn in desc_force:
                tmpl = DESC_TEMPLATES[dn]
                vi = min(desc_force[dn], len(tmpl) - 1)
                text = tmpl[vi][1]
            else:
                text = pick_desc(dn, rng)
            desc_rows.append({
                "app_no": app_no, "detail_id": did,
                "res_result": text, "res_date": app_time,
            })
            item_rows.append({
                "detail_id": did, "det_time": app_time, "app_no": app_no,
                "pat_id": pat_id, "det_com_id": "", "det_com_name": dn,
                "det_dep_id": "", "det_dep_name": DEPT_MAP.get(dn, "Health Examination Center"),
                "det_price": "0.00", "det_amount": "1.0", "det_discount": "1.0",
                "det_totalprice": "0.00", "det_discount_totalprice": "0.00",
                "det_charge_flag": "1", "det_charge_type": "1", "det_rep_type": "0",
                "det_userid": "demo", "det_username": "demo", "det_usercode": "demo",
                "det_flag": "0", "det_signin_flag": "0", "det_append_flag": "0",
                "det_aba_flag": "0", "det_delay_flag": "0", "det_apply_sendflag": "0",
                "det_del_flag": "0", "det_balance_flag": "0", "det_restype": "0",
                "det_add_item": "0", "det_checkself": "0", "det_fee_type": "0",
                "det_extend_flag": "0",
            })

        # --- Summary
        summary_rows.append({
            "detail_id": str(detail_id), "app_no": app_no,
            "sum_summary": "All examinations in this health check are complete; see the individual reports.",
            "sum_advice": "", "sum_advice_spec": "", "sum_create_time": app_time,
            "sum_create_userid": "demo", "sum_create_usercode": "demo",
            "sum_create_username": DOCTOR_NAME, "sum_exam_time": app_time,
            "sum_exam_userid": "demo", "sum_exam_usercode": "demo",
            "sum_exam_username": DOCTOR_NAME, "sum_audit_time": "",
            "sum_audit_userid": "", "sum_audit_usercode": "", "sum_audit_username": "",
        })

        if (i + 1) % 1000 == 0:
            print(f"   … {i + 1}/{count}")

    os.makedirs(OUT_DIR, exist_ok=True)
    print("3) Writing CSVs...")
    tables = {
        "PES_APPLY.csv": (apply_rows, COLS_APPLY),
        "PES_PATIENT.csv": (patient_rows, COLS_PATIENT),
        "PES_RESULT.csv": (result_rows, COLS_RESULT),
        "PES_apply_item.csv": (item_rows, COLS_ITEM),
        "PES_RESULT_DESC.csv": (desc_rows, COLS_DESC),
        "PES_SUMMARY_COMBINE.csv": (summary_rows, COLS_SUMMARY),
    }
    for fname, (rows, cols) in tables.items():
        df = pd.DataFrame(rows, columns=cols)
        path = os.path.join(OUT_DIR, fname)
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"   {fname:28s} {len(df):>8d} rows  {os.path.getsize(path)/1e6:6.1f} MB")

    manifest = {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "generator": "tools/build_snapshot.py",
        "seed": seed,
        "patient_count": count,
        "app_no_range": [str(APP_NO_START), str(APP_NO_START + count - 1)],
        "pat_id_range": [f"{PAT_ID_PREFIX}{1:06d}", f"{PAT_ID_PREFIX}{count:06d}"],
        "lab_item_count": len(profile),
        "desc_item_count": len(desc_names),
        "policy": {
            "app_no": "re-generated, consecutive from 90000001",
            "pat_id": "re-generated, from P900000001",
            "pat_name/app_pat_name": "SubjectNNNN",
            "pat_address": "No. N, Virtual Street, Demo District, Demo City, Demo Province",
            "pat_nation": "constant Han",
            "pat_idcard/pat_mobile/pat_email/pat_card_no/pat_contact/pat_photo": "all left blank",
            "lab_values": "resampled from item distributions learned from historical data; no real patient record is copied",
            "desc_text": "built-in template library; no real report text is quoted",
        },
    }
    with open(os.path.join(OUT_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"4) Done -> {OUT_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20261110)
    args = ap.parse_args()
    build(args.count, args.seed)
