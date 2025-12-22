'''
  Ingest data for RetailBoost - MULTI-FOLDER SCAN AND FAST LOAD ANALYTICS FIXES
'''

import json
import pandas as pd
import psycopg2
import psycopg2.extras
import os
import requests
from dotenv import load_dotenv
import re
from pathlib import Path
import gc
from io import StringIO

from scripts.utils import json_serial
from scripts.constants import MAX_COLUMN_NAME_LENGTH
from scripts.analytics import generate_retail_insights 

load_dotenv()

DB_URL = os.getenv("DATABASE_URL")

# Define all folders where data might reside
DATA_DIRS = [Path('data'), Path('output')]

# Path to google drive for large CSVs https://drive.google.com/uc?id=1-eIwvMTWp3SreJrsRe3BochHP_QFxgxH&export=download
# Google Drive Configuration
GDRIVE_FILE_ID = "1-eIwvMTWp3SreJrsRe3BochHP_QFxgxH"
# Name the file whatever the table should be named in Postgres (e.g., hpg_reserve.csv or horizon42.csv)
GDRIVE_FILENAME = "hpg_reserve.csv" 


DATE_COLUMNS = {"visit_date", "visit_datetime", "reserve_datetime"}
BATCH_SIZE = 5000 

def filename_to_table(filename):
    # Normalize filename to table name (lowercase, no extension)
    return re.sub(r'\.csv$', '', filename).lower()

def infer_sql_type(series: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(series): return "TIMESTAMP"
    elif pd.api.types.is_integer_dtype(series): return "INTEGER"
    elif pd.api.types.is_float_dtype(series): return "FLOAT"
    else: return "TEXT"

def clean_dataframe(df):
    for col in df.columns:
        if col in DATE_COLUMNS:
            try: df[col] = pd.to_datetime(df[col], errors="coerce")
            except Exception: pass
    df = df.where(pd.notnull(df), None)
    return df

def download_from_google_drive(file_id, destination):
    print(f"   -> Downloading file from Google Drive to {destination}...", flush=True)
    url = f'https://drive.google.com/uc?id={file_id}&export=download'
    
    try:
        response = requests.get(url, stream=True)
        # Handle the "Google Virus Scan" warning page for large files
        if "confirm" in response.cookies:
            url = f"{url}&confirm={response.cookies['confirm']}"
            response = requests.get(url, stream=True)
            
        if response.status_code == 200:
            with open(destination, 'wb') as f:
                for chunk in response.iter_content(chunk_size=32768):
                    if chunk:
                        f.write(chunk)
            print("   -> Download complete.", flush=True)
        else:
            print(f"   -> Failed to download. Status code: {response.status_code}", flush=True)
    except Exception as e:
        print(f"   -> Download Error: {e}", flush=True)
# -----------------------------------------------------

def update_task_db(task_id, status, progress=None, result=None):
    conn = psycopg2.connect(DB_URL)
    try:
        with conn.cursor() as cur:
            progress_json = json.dumps(progress, default=json_serial) if progress else None
            result_json = json.dumps(result, default=json_serial) if result else None
            sql = """
                INSERT INTO task_status (task_id, status, progress, result)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (task_id) DO UPDATE SET 
                    status = EXCLUDED.status, progress = EXCLUDED.progress, result = EXCLUDED.result;
            """
            cur.execute(sql, (task_id, status, progress_json, result_json))
        conn.commit()
    except Exception as e:
        print(f"Error updating task DB: {e}", flush=True)
    finally:
        conn.close()

def create_table_from_header(conn, filepath, table_name):
    df_header = pd.read_csv(filepath, nrows=100)
    df_header.columns = [c[:MAX_COLUMN_NAME_LENGTH] for c in df_header.columns]
    cols_sql = [f'"{col}" {infer_sql_type(df_header[col])}' for col in df_header.columns]
    
    with conn.cursor() as cur:
        cur.execute(f'CREATE TABLE IF NOT EXISTS "{table_name}" ({", ".join(cols_sql)});')
    conn.commit()

def drop_table_if_exists(conn, table_name):
    with conn.cursor() as cur:
        cur.execute(f'DROP TABLE IF EXISTS "{table_name}" CASCADE;')
    conn.commit()

def fast_load_csv(conn, filepath, table_name, task_id):
    print(f"   -> Fast Load: Processing {table_name} (from {filepath})...", flush=True)
    drop_table_if_exists(conn, table_name)
    create_table_from_header(conn, filepath, table_name)
    
    total_rows = 0
    try:
        with open(filepath, 'r') as f: total_file_rows = sum(1 for _ in f) - 1
        
        with conn.cursor() as cur:
            for i, chunk in enumerate(pd.read_csv(filepath, chunksize=50000)):
                chunk = clean_dataframe(chunk)
                output = StringIO()
                chunk.to_csv(output, sep='\t', header=False, index=False)
                output.seek(0)
                
                cur.copy_expert(f'COPY "{table_name}" FROM STDIN WITH (FORMAT CSV, DELIMITER E\'\\t\', NULL \'\')', output)
                conn.commit()
                
                rows_in_chunk = len(chunk)
                total_rows += rows_in_chunk
                
                if task_id:
                    update_task_db(task_id, "processing", progress={
                        "message": f"Loading {table_name}",
                        "current": total_rows,
                        "total": total_file_rows,
                        "percent": int((total_rows / total_file_rows) * 100) if total_file_rows > 0 else 0
                    })
                
                del chunk, output
                gc.collect()
                
    except Exception as e:
        print(f"   -> Fast Load Error on {table_name}: {e}", flush=True)
        conn.rollback()
        raise e

def fetch_all_tables_preview(table_names: list[str], limit: int = None):
    results = {}
    conn = psycopg2.connect(DB_URL)
    try:
        with conn.cursor() as cur:
            for table in table_names:
                try:
                    if limit is None:
                        # ANALYTICS FETCH LOGIC
                        if table == 'air_visit_data':
                            print(f"Fetching SAFE subset of {table} (Limit 50k)...", flush=True)
                            query = f'SELECT air_store_id, visit_date, CAST(visitors AS INTEGER) as visitors FROM "{table}" ORDER BY visit_date DESC LIMIT 50000;'
                            cur.execute(query)
                        elif table == 'air_store_info':
                            cur.execute(f'SELECT air_store_id, air_genre_name, air_area_name FROM "{table}";')
                        elif 'horizon42' in table:
                            print(f"Fetching Forecast Table: {table}...", flush=True)
                            cur.execute(f'SELECT * FROM "{table}";')
                        else:
                            continue
                    else:
                        cur.execute(f'SELECT * FROM "{table}" LIMIT {limit};')
                    
                    rows = cur.fetchall()
                    if cur.description: col_names = [desc[0] for desc in cur.description]
                    else: col_names = []

                    if limit is None:
                        results[table] = pd.DataFrame(rows, columns=col_names) if rows else pd.DataFrame(columns=col_names)
                    else:
                        results[table] = [dict(zip(col_names, row)) for row in rows]
                    
                    del rows
                    gc.collect()

                except Exception as inner_e:
                    print(f"Error fetching {table}: {inner_e}", flush=True)
                    pass
    finally:
        conn.close()
    return results

def check_table_exists(conn, table_name):
    with conn.cursor() as cur:
        cur.execute(f"SELECT to_regclass('public.\"{table_name}\"');")
        return cur.fetchone()[0] is not None

def process_all_csvs(task_id, table_name):
    print(f"Processing started! Task ID: {task_id}", flush=True)
    update_task_db(task_id, "processing", progress={"message": "Initializing...", "percent": 0})
    
    try:
        conn = psycopg2.connect(DB_URL)

        # 0. ENSURE DATA DIRECTORY EXISTS AND DOWNLOAD REMOTE CSV
        Path('data').mkdir(parents=True, exist_ok=True)
        
        if GDRIVE_FILE_ID:
            local_dest = Path('data') / GDRIVE_FILENAME
            # Only download if not exists (or always overwrite if you prefer fresh data)
            # On Heroku, 'data' is wiped on restart, so this will always download fresh.
            download_from_google_drive(GDRIVE_FILE_ID, local_dest)
        
        # 1. GATHER FILES FROM DATA AND OUTPUT FOLDERS
        csv_files = []
        for d in DATA_DIRS:
            if d.exists():
                csv_files.extend(list(d.glob("*.csv")))
            else:
                print(f"Warning: Directory {d} not found.", flush=True)

        if not csv_files:
            print("No CSV files found in data/ or output/ directories.", flush=True)

        # 2. FAST INGESTION
        processed_tables_current_run = []
        for i, filepath in enumerate(csv_files):
            t_name = filename_to_table(filepath.name)
            print(f"******* Processing {filepath.name} -> Table: {t_name}", flush=True)
            
            fast_load_csv(conn, filepath, t_name, task_id)
            processed_tables_current_run.append(t_name)
            gc.collect()
        conn.close()

        # 3. PREVIEWS
        print("Fetching previews...", flush=True)
        conn_prev = psycopg2.connect(DB_URL)
        partial_results = {}
        for t in processed_tables_current_run:
            try:
                df_prev = pd.read_sql(f'SELECT * FROM "{t}" LIMIT 10', conn_prev)
                partial_results[t] = df_prev.to_dict(orient='records')
            except: pass
        conn_prev.close()

        # 4. ANALYTICS
        print("Running Analytics Engine...", flush=True)
        conn = psycopg2.connect(DB_URL)
        
        forecast_table_name = None
        
        # Priority 1: Check current run for forecast
        for t in processed_tables_current_run:
            if "sub_forecast_horizon42" in t:
                forecast_table_name = t
                print(f"   -> Found Forecast Table in current run: {t}", flush=True)
                break
        
        # Priority 2: Check DB for existing forecast
        if not forecast_table_name:
            with conn.cursor() as cur:
                cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public' AND table_name LIKE '%horizon42%';")
                res = cur.fetchone()
                if res: 
                    forecast_table_name = res[0]
                    print(f"   -> Found Forecast Table in DB: {forecast_table_name}", flush=True)
        
        conn.close()

        tables_to_fetch = ['air_visit_data', 'air_store_info']
        if forecast_table_name: tables_to_fetch.append(forecast_table_name)
        if check_table_exists(psycopg2.connect(DB_URL), 'date_info'): tables_to_fetch.append('date_info')

        analytics_dfs = fetch_all_tables_preview(tables_to_fetch, limit=None)
        
        if forecast_table_name and forecast_table_name in analytics_dfs:
            analytics_dfs['forecast_data'] = analytics_dfs.pop(forecast_table_name)
        else:
            print("   -> WARNING: Forecast table identified but data not loaded into analytics_dfs", flush=True)

        chart_data = generate_retail_insights(analytics_dfs)
        
        del analytics_dfs
        gc.collect()

        # 5. SAVE FINAL RESULT
        final_result = partial_results.copy()
        if chart_data and "generated_charts" in chart_data:
            print("   -> Merging Charts into Final Result", flush=True)
            final_result["chart_data"] = chart_data
        else:
            print("   -> WARNING: chart_data is empty or missing keys", flush=True)

        update_task_db(task_id, "completed", progress={"percent": 100, "message": "Done"}, result=final_result)
        print(f"Task {task_id} completed successfully.")
        
    except Exception as e:
        print(f"Error processing CSVs: {e}", flush=True)
        update_task_db(task_id, "failed", result={"error": str(e)})
