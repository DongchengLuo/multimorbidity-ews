# db_fetcher.py
import pandas as pd
import os
from typing import List, Dict
from path_utils import ensure_dir_exists

# Snapshot mode: no database; read directly from data/snapshot/
SNAPSHOT_MODE = os.getenv("SNAPSHOT_MODE", "1").strip().lower() not in (
    "0", "false", "no", "off", ""
)


class DatabaseFetcher:
    def __init__(self, server: str, db_name: str, user: str, password: str):
        self.server = server
        self.db_name = db_name
        self.user = user
        self.password = password

    def fetch_patient_data(self, app_nos: List[str], output_dir: str) -> Dict:
        """
        Fetch patient data from the database and save it to the given directory
        Returns: {table_name: file_path}
        """
        app_nos = [str(value).strip() for value in app_nos if str(value).strip()]
        if not app_nos:
            raise ValueError("app_nos must not be empty")

        ensure_dir_exists(output_dir)

        if SNAPSHOT_MODE:
            return self._fetch_from_snapshot(app_nos, output_dir)

        tables = [
            'dbo.PES_APPLY',
            'dbo.PES_PATIENT',
            'dbo.PES_RESULT',
            'dbo.PES_apply_item',
            'dbo.PES_SUMMARY_COMBINE',
            'dbo.PES_RESULT_DESC'
        ]

        # app_no is passed only as a query parameter to avoid splicing web input into SQL.
        placeholders = ",".join(["%s"] * len(app_nos))
        file_paths = {}

        try:
            import pymssql  # lazy import; not needed in snapshot mode
            conn = pymssql.connect(
                server=self.server, user=self.user, password=self.password,
                database=self.db_name, charset='utf8'
            )

            for table in tables:
                file_name = f"{table.split('.')[-1]}.csv"
                save_path = os.path.join(output_dir, file_name)
                file_paths[table] = save_path

                if 'PES_PATIENT' in table:
                    sql = f"SELECT * FROM {table} WHERE pat_id IN (SELECT pat_id FROM dbo.PES_APPLY WHERE app_no IN ({placeholders}))"
                else:
                    sql = f"SELECT * FROM {table} WHERE app_no IN ({placeholders})"

                df = pd.read_sql(sql, conn, params=app_nos)

                if not df.empty:
                    # Data-cleaning logic (extracted from the original web script)
                    def fix_mojibake(text):
                        if isinstance(text, str):
                            try:
                                return text.encode('latin-1').decode('gbk')
                            except Exception:
                                return text
                        return text

                    for col in df.select_dtypes(include=['object', 'string']).columns:
                        df[col] = df[col].apply(fix_mojibake)

                    if 'pat_name' in df.columns:
                        df['pat_name'] = df['pat_name'].astype(str).str.strip()
                        df['pat_name'] = df['pat_name'].str.replace('\t', '', regex=False)
                        df['pat_name'] = df['pat_name'].str.replace('\x00', '', regex=False)

                    df.to_csv(save_path, index=False, encoding='utf-8-sig')
                    print(f"[OK] Fetched {len(df)} rows -> {file_name}")
                else:
                    print(f"[WARN] Table {table} has no data")

            conn.close()
            return file_paths

        except Exception as e:
            print(f"[ERROR] Database fetch failed: {e}")
            raise

    def _fetch_from_snapshot(self, app_nos: List[str], output_dir: str) -> Dict:
        """Snapshot mode: read from the local de-identified snapshot and persist; no database connection at all."""
        import snapshot_store

        tables = [
            'dbo.PES_APPLY',
            'dbo.PES_PATIENT',
            'dbo.PES_RESULT',
            'dbo.PES_apply_item',
            'dbo.PES_SUMMARY_COMBINE',
            'dbo.PES_RESULT_DESC',
        ]
        data = snapshot_store.fetch_tables(app_nos)
        file_paths = {}

        for table in tables:
            file_name = f"{table.split('.')[-1]}.csv"
            save_path = os.path.join(output_dir, file_name)
            file_paths[table] = save_path

            if 'PES_SUMMARY_COMBINE' in table:
                df = data.get(file_name)
            else:
                df = data.get(file_name)

            if df is not None and not df.empty:
                df.to_csv(save_path, index=False, encoding='utf-8-sig')
                print(f"[OK] [Snapshot] {len(df)} rows -> {file_name}")
            else:
                # Match real mode: create empty tables even without data, so downstream reads do not fail
                if df is not None:
                    df.to_csv(save_path, index=False, encoding='utf-8-sig')
                print(f"[WARN] [Snapshot] Table {table} has no data")

        return file_paths
