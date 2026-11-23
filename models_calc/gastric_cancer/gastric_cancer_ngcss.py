import pandas as pd
import re
import os
import warnings

# Ignore DtypeWarning
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)


class NGCSSGastricModel:
    """
    New gastric cancer screening score (NGCSS)
    Implemented per Gut 2019;68:1576-1587
    """
    # 1. Field-mapping table (for exact matching)
    FIELD_MAPPING = {
        'age': ['app_pat_age', 'age'],
        'gender': ['pat_sex', 'sex', 'gender'],
        'pgr': ['pepsinogen_I_II', 'pgr', 'PGR', 'pepsinogen_ratio'],
        'g17': ['gastrin_17', 'g17', 'G17'],
        'hp_positive': ['h_pylori_serology', 'h_pylori_serology_monitoring', 'Hp'],
        'regular_pickled': ['frequent_salt_preserved_food', 'regular_pickled'],
        'regular_fried': ['frequent_fried_food', 'regular_fried'],
        'history_labels': ['past_history', 'past_medical_history', 'history_labels']
    }

    def __init__(self):
        pass

    def predict(self, raw_data):
        try:
            # 1. Required core-indicator check (any missing item blocks computation)
            required = ['age', 'gender', 'pgr', 'g17', 'hp_positive']
            for k in required:
                val = raw_data.get(k)
                if pd.isna(val) or val == '':
                    return None, None, "-", f"scoring aborted due to missing core indicator [{k}]"

            # 2. Numeric parsing and basic handling
            try:
                age = float(raw_data['age'])
                pgr = float(raw_data['pgr'])
                g17 = float(raw_data['g17'])
            except (ValueError, TypeError):
                return None, None, "-", "numeric parse error; non-numeric abnormal data present"

            # Force exact sex matching
            gender_val = str(raw_data['gender']).strip().lower()
            is_male = gender_val in ['male', '1', 'm']

            # 3. Smart parsing of text/boolean labels (exact-match logic)
            def parse_label_exact(val, pos_list):
                if pd.isna(val) or val == '': return 0
                s = str(val).strip().lower()
                return 1 if s in pos_list else 0

            hp_pos = parse_label_exact(raw_data.get('hp_positive'), ['positive', '1', 'yes', 'true', '+'])
            pickled = parse_label_exact(raw_data.get('regular_pickled'), ['often', '1', 'yes', 'true'])
            fried = parse_label_exact(raw_data.get('regular_fried'), ['often', '1', 'yes', 'true'])

            # Past-history check (split by delimiters, then exact match)
            history = str(raw_data.get('history_labels', '')).strip()
            history_list = [h.strip() for h in re.split(r'[,，、;；]', history) if h.strip()]
            if 'gastric_cancer' in history_list:
                return None, None, "-", "excluded: prior gastric cancer history"

            # 4. Run the NGCSS scoring logic
            score = 0

            # Age scoring
            if 50 <= age <= 59:
                score += 4
            elif 60 <= age <= 69:
                score += 6
            elif age >= 70:
                score += 9

            # Sex scoring
            if is_male: score += 4

            # Dietary habits
            if pickled: score += 2
            if fried: score += 2

            # Helicobacter pylori
            if hp_pos: score += 1

            # PGR (PG I/II ratio)
            if pgr < 3.89: score += 3

            # G-17 level
            if 1.50 <= g17 <= 5.70:
                score += 3
            elif g17 > 5.70:
                score += 4

            # 5. Risk tiering and code definitions
            if 0 <= score <= 11:
                level = "low risk"
                advice = "current risk is low; recheck relevant serology periodically."
                code = "C16.1.1"
            elif 12 <= score <= 16:
                level = "medium risk"
                advice = "some risk present; gastroscopy advised to exclude lesions."
                code = "C16.1.2"
            else:
                level = "high risk"
                advice = "risk markedly elevated; strongly recommend prompt gastroscopy."
                code = "C16.1.3"

            return score, code, level, advice

        except Exception as e:
            return None, None, "-", f"error: {str(e)}"


class BatchProcessor:
    def __init__(self):
        self.model = NGCSSGastricModel()
        # Invert the mapping for fast lookup of an item's internal key (e.g. 'pepsinogen I/II' -> 'pgr')
        self.item_to_key = {}
        for k, aliases in self.model.FIELD_MAPPING.items():
            for a in aliases:
                self.item_to_key[a] = k

    def run(self, base_path, exam_path, output_path):
        print("--- Starting gastric cancer NGCSS model ---")
        print("loading data...")

        # Ported safe-read function
        def read_csv_safe(path):
            for enc in ['gbk', 'utf-8-sig', 'utf-8']:
                try:
                    return pd.read_csv(path, encoding=enc, low_memory=False)
                except:
                    continue
            return pd.DataFrame()

        base_df = read_csv_safe(base_path)
        exam_df = read_csv_safe(exam_path)

        if base_df.empty or exam_df.empty:
            print(" failed to read CSV or file is empty")
            return

        # ---------------------------------------------------------
        # 1. Preprocessing and wide-table pivot
        # ---------------------------------------------------------
        print("pivoting exam data...")

        # Trim spaces (infer first, then stringify, to normalize leading zeros)
        exam_df['app_no'] = exam_df['app_no'].astype(str).str.strip()
        exam_df['pat_id'] = exam_df['pat_id'].astype(str).str.strip()
        base_df['app_no'] = base_df['app_no'].astype(str).str.strip()
        base_df['pat_id'] = base_df['pat_id'].astype(str).str.strip()

        # Tag each row with a model category label
        exam_df['cat'] = exam_df['res_itm_name'].str.strip().map(self.item_to_key)

        # Keep only the indicators the model needs
        filtered = exam_df.dropna(subset=['cat']).copy()

        records = []
        # Group and aggregate by order number + patient ID
        for (app_no, pat_id), group in filtered.groupby(['app_no', 'pat_id']):
            row = {'app_no': app_no, 'pat_id': pat_id}
            for _, item in group.iterrows():
                cat = item['cat']
                if cat not in row:
                    row[cat] = item['res_value']
            records.append(row)

        wide_items = pd.DataFrame(records)

        if wide_items.empty:
            print(" no relevant exam indicators found (PGR/G17/Hp).")
            return

        # ---------------------------------------------------------
        # 2. Merge the base table with the exam wide table
        # ---------------------------------------------------------
        print("merging base info with exam data...")
        final = pd.merge(wide_items, base_df, on=['app_no', 'pat_id'], how='inner')

        if final.empty:
            print(" merge failed: no matching intersection on app_no and pat_id.")
            return

        # ---------------------------------------------------------
        # 3. Compute risk scores
        # ---------------------------------------------------------
        print(f"merge succeeded; computing {len(final)} valid records...")
        results = []
        GAUGE_REF = "0-11|12-16|17-25"

        for _, r in final.iterrows():
            age_val = r.get('app_pat_age') if pd.notna(r.get('app_pat_age')) else r.get('age')
            gender_val = r.get('pat_sex') if pd.notna(r.get('pat_sex')) else r.get('sex')
            history_val = r.get('past_history') if pd.notna(r.get('past_history')) else r.get('past_history')

            data = {
                'age': age_val,
                'gender': gender_val,
                'name': r.get('pat_name'),
                'history_labels': history_val,
                'pgr': r.get('pgr'),
                'g17': r.get('g17'),
                'hp_positive': r.get('hp_positive')
            }

            score, code, level, advice = self.model.predict(data)

            output_row = {
                'pat_id': r['pat_id'],
                'app_no': r['app_no'],
                'pat_name': data.get('name', 'missing'),
                'age': data.get('age', 'missing'),
                'gender': data.get('gender', 'missing'),
                'code': code if code is not None else 'N/A',
                'predicted_score': score if score is not None else 'missing',
                'risk_level': level,
                'prediction_horizon': 'current',
                'model_name': 'Gastric Cancer NGCSS Screening Score',
                'gauge_reference': GAUGE_REF,
                'pepsinogen_ratio': data.get('pgr', 'missing'),
                'gastrin_17': data.get('g17', 'missing'),
                'Hp': data.get('hp_positive', 'missing'),
            }
            results.append(output_row)

        # ---------------------------------------------------------
        # 4. Export results
        # ---------------------------------------------------------
        if results:
            df_final = pd.DataFrame(results)
            df_final.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f" computation complete: {len(df_final)} results.")
            print(f"[OK] file saved to: {os.path.abspath(output_path)}")
        else:
            print(" no valid data generated.")


if __name__ == "__main__":
    BatchProcessor().run(
        "base.csv",
        "item_result.csv",
        "ngcss_gastric_result.csv"
    )