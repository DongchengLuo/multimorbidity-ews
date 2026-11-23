import pandas as pd
import re
import os
import warnings
import numpy as np

# Suppress unnecessary dtype warnings
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)

class DiabetesNCDRSModel:
    MODEL_NAME = "New China Diabetes Risk Score (NCDRS)"
    # Three-band gauge reference (low|medium|high)
    GAUGE_REFERENCE = "0-22|23-28|29-51"

    # Exact-match field dict (adds height and weight)
    FIELD_MAPPING = {
        'age': ['app_pat_age', 'age'],
        'gender': ['pat_sex', 'sex', 'gender'],
        'BMI': ['bmi', 'BMI', '(body mass index) bmi'],
        'height': ['height'],      # added height mapping
        'weight': ['weight'],      # added weight mapping
        'wc': ['waist', 'wc'],
        'sbp': ['blood_pressure', 'systolic_bp', 'sbp'],
        'diabetes_family_history': ['diabetes_family_history', 'first_degree_diabetes_history', 'family_history'],
        'has_diabetes': ['past_history', 'past_medical_history', 'diabetes'],
    }

    def predict(self, raw_data):
        """Run the NCDRS scoring logic"""
        try:
            # 1. Core indicator check (return a missing notice immediately if any core indicator is absent; do not compute)
            required = ['age', 'gender', 'BMI', 'wc', 'sbp']
            for k in required:
                if raw_data.get(k) is None or pd.isna(raw_data.get(k)): 
                    return "-", "-", "-" # return '-' when the model result is missing

            # 2. Past-history parsing (detect negation)
            def parse_history(val, keyword):
                if pd.isna(val): return 0
                s = str(val)
                if any(neg in s for neg in ['absent', 'no', 'no disease', 'non-smoker', 'refuses']): return 0
                if keyword in s: return 1
                return 1 if s.strip() in ['1', 'yes', 'True'] else 0

            if parse_history(raw_data.get('has_diabetes'), 'diabetes') == 1:
                return "-", "-", "-" # return '-' when already diseased

            # 3. Numeric parsing
            age = float(raw_data['age'])
            gender_val = str(raw_data['gender'])
            is_male = any(x in gender_val for x in ['male', '1'])
            bmi = float(raw_data['BMI'])
            wc = float(raw_data['wc'])
            sbp_raw = str(raw_data['sbp'])
            
            # Special handling for blood pressure: extract systolic (high) pressure
            try:
                nums = re.findall(r"(\d+\.?\d*)", sbp_raw)
                nums = [float(x) for x in nums]
                sbp = max(nums[:2]) if len(nums) >= 2 else nums[0]
            except:
                 return "-", "-", "-"

            family_history = parse_history(raw_data.get('diabetes_family_history'), 'diabetes')

            # 4. Run the scoring logic
            score = 0
            # Age points
            if age < 25: score += 0
            elif age <= 34: score += 4
            elif age <= 39: score += 8
            elif age <= 44: score += 11
            elif age <= 49: score += 12
            elif age <= 54: score += 13
            elif age <= 59: score += 15
            elif age <= 64: score += 16
            else: score += 18
            
            # Sex points
            if is_male: score += 2
            
            # BMI points
            if bmi < 22: score += 0
            elif bmi < 24: score += 1
            elif bmi < 30: score += 3
            else: score += 5
            
            # Waist-circumference points
            if is_male:
                if wc < 75: score += 0
                elif wc < 80: score += 3
                elif wc < 85: score += 5
                elif wc < 90: score += 7
                elif wc < 95: score += 8
                else: score += 10
            else:
                if wc < 70: score += 0
                elif wc < 75: score += 3
                elif wc < 80: score += 5
                elif wc < 85: score += 7
                elif wc < 90: score += 8
                else: score += 10
            
            # Blood-pressure points
            if sbp < 110: score += 0
            elif sbp < 120: score += 1
            elif sbp < 130: score += 3
            elif sbp < 140: score += 6
            elif sbp < 150: score += 7
            elif sbp < 160: score += 8
            else: score += 10
            
            # Family history
            if family_history: score += 6

            # 5. Result tiering
            if score <= 22:
                risk_level = "low risk"
                code = "E14.2.1"
            elif 23 <= score <= 28:
                risk_level = "medium risk"
                code = "E14.2.2"
            else:
                risk_level = "high risk"
                code = "E14.2.3"

            return score, risk_level, code

        except Exception as e:
            return "-", "-", "-"


class BatchProcessor:
    def __init__(self):
        self.model = DiabetesNCDRSModel()
        self.item_to_key = {}
        # Exact-match dict mapping
        for k, aliases in self.model.FIELD_MAPPING.items():
            for a in aliases: 
                # Lowercase and trim for later matching
                self.item_to_key[re.sub(r'[\(\)\（\）\[\]\【\】\{\}\s]', '', str(a)).lower()] = k

    def read_csv_safe(self, path, **kwargs):
        """Robust reference reading logic"""
        for enc in ['utf-8-sig', 'gbk', 'gb18030', 'utf-8']:
            try:
                df = pd.read_csv(path, encoding=enc, low_memory=False, **kwargs)
                print(f" successfully read {path} (encoding: {enc})")
                return df
            except:
                continue
        print(f" cannot read file: {path}")
        return pd.DataFrame()

    def run(self, base_path, exam_path, output_path):
        """Batch-process data"""
        # Coerce to string so int values do not break lstrip
        base_df = self.read_csv_safe(base_path, dtype=str)
        exam_df = self.read_csv_safe(exam_path, dtype=str)

        if base_df.empty:
            print(" base data empty or unreadable; task aborted.")
            return

        # ================= [Key fix] =================
        # Strip leading '0' from app_no and pat_id for perfect matching
        base_df['app_no'] = base_df['app_no'].astype(str).str.strip().str.lstrip('0')
        base_df['pat_id'] = base_df['pat_id'].astype(str).str.strip().str.lstrip('0')
        # ==============================================

        # Pivot exam indicators to a wide table
        records = []
        if not exam_df.empty:
            # Also drop useless '0' values from the exam detail table
            exam_df['app_no'] = exam_df['app_no'].astype(str).str.strip().str.lstrip('0')
            exam_df['pat_id'] = exam_df['pat_id'].astype(str).str.strip().str.lstrip('0')
            
            for (pid, app_no), group in exam_df.groupby(['pat_id', 'app_no']):
                data = {'pat_id': pid, 'app_no': app_no}
                
                # Iterate details with exact matching and first-value-only logic
                for _, row in group.iterrows():
                    if pd.isna(row.get('res_itm_name')): continue
                    itm_raw = str(row['res_itm_name'])
                    itm_norm = re.sub(r'[\(\)\（\）\[\]\【\】\{\}\s]', '', itm_raw).lower()
                    
                    if itm_norm in self.item_to_key:
                        standard_key = self.item_to_key[itm_norm]
                        # Take the first value encountered
                        if standard_key not in data or pd.isna(data[standard_key]):
                            data[standard_key] = row['res_value']
                records.append(data)
                
        wide_items = pd.DataFrame(records) if records else pd.DataFrame(columns=['app_no', 'pat_id'])
        
        # Left join keeping everyone in base_df
        final_merged = pd.merge(base_df, wide_items, on=['app_no', 'pat_id'], how='left')

        results = []
        for _, row in final_merged.iterrows():
            # Fetch age and sex (prefer the base table)
            age = row.get('app_pat_age') if pd.notna(row.get('app_pat_age')) else row.get('age')
            gender = row.get('pat_sex') if pd.notna(row.get('pat_sex')) else row.get('gender')
            
            # Assemble the data dict passed to the model
            data = {
                'age': age,
                'gender': gender,
                'BMI': row.get('BMI'),
                'height': row.get('height'),
                'weight': row.get('weight'),
                'wc': row.get('wc'),
                'sbp': row.get('sbp'),
                'diabetes_family_history': row.get('diabetes_family_history'),
                'has_diabetes': row.get('has_diabetes')
            }

            # ================= BMI fallback computation =================
            if pd.isna(data.get('BMI')):
                if pd.notna(data.get('height')) and pd.notna(data.get('weight')):
                    try:
                        h_val = float(re.sub(r'[^\d.]', '', str(data['height'])))
                        w_val = float(re.sub(r'[^\d.]', '', str(data['weight'])))
                        h_m = h_val / 100.0 if h_val > 3.0 else h_val
                        if h_m > 0:
                            data['BMI'] = round(w_val / (h_m ** 2), 2)
                    except Exception:
                        pass
            # ========================================================

            # Run prediction
            score, level, code = self.model.predict(data)

            # Format input display; show "missing" when absent
            def fmt_in(v): return "missing" if pd.isna(v) or v is None or str(v).strip() == "" else v

            output_row = {
                'pat_id': row['pat_id'],
                'app_no': row['app_no'],
                'pat_name': row.get('pat_name', 'unknown'),
                'model_name': self.model.MODEL_NAME,
                # --- Raw indicator section ---
                'age': fmt_in(data.get('age')),
                'sex': fmt_in(data.get('gender')),
                'height': fmt_in(data.get('height')),     
                'weight': fmt_in(data.get('weight')),     
                'BMI': fmt_in(data.get('BMI')),
                'waist': fmt_in(data.get('wc')),
                'blood_pressure': fmt_in(data.get('sbp')),
                'family_history': fmt_in(data.get('diabetes_family_history')),
                # --- Model result section ---
                'predicted_score': score,
                'risk_level': level,
                'prediction_horizon': 'next 3 years' if score != "-" else "-",
                'code': code,
                'gauge_reference': self.model.GAUGE_REFERENCE if score != "-" else "-"
            }
            results.append(output_row)

        if results:
            df_final = pd.DataFrame(results)
            df_final.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f" computation complete: {len(results)} results to {output_path}")

if __name__ == "__main__":
    processor = BatchProcessor()
    processor.run(
        "base.csv",
        "item_result.csv",
        "diabetes_risk_ncdrs_final.csv"
    )
