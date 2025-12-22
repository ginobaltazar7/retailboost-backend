# Standard library
import asyncio
import uuid
import psycopg2
import os

# 3rd party
from fastapi import APIRouter, Query, BackgroundTasks
from fastapi import HTTPException
from fastapi.responses import (
    JSONResponse,
    FileResponse
)


# local
from models.schemas import DataOutput
from scripts.data_pipeline import process_all_csvs, DB_URL

router = APIRouter()


@router.post("/ingest", response_model=DataOutput)
async def ingest(
    background_tasks: BackgroundTasks,
    table_name: str = Query(
        'air_reserve', description='Table to fetch data from after processing')
):
    print("Starting CSV processing...")
    # Generate a new task ID for this batch
    task_id = str(uuid.uuid4())

    # try:
    # Trigger CSV processing in background thread
    # await asyncio.to_thread(process_all_csvs)
    background_tasks.add_task(process_all_csvs, task_id, table_name)
    # Optionally, you can trigger fetching data after some delay or manually later
    # For now, just confirm processing has started

    # except Exception as e:
    #    print(f"Error during CSV processing: {e}")
    #    raise

    # Fetch first 10 rows from the specified table
    # first_rows = await asyncio.to_thread(fetch_first_rows, table_name)
    # background_tasks.add_task(fetch_first_rows, table_name)
    return {
        "message": "CSV processing started.",
        "task_id": task_id,
    }


@router.get("/results/{task_id}")
async def get_results(task_id: str):
    conn = psycopg2.connect(DB_URL)
    task_info = None

    try:
        # Fetch from DB instead of memory
        with conn.cursor() as cur:
            cur.execute("""
                SELECT status, progress, result 
                FROM task_status 
                WHERE task_id = %s
            """, (task_id,))
            row = cur.fetchone()

            if row:
                task_info = {
                    "status": row[0],
                    "progress": row[1],
                    "result": row[2]
                }
    except Exception as e:
        print(f"DB Error: {e}")
        raise HTTPException(
            status_code=500, detail="Database connection error")
    finally:
        conn.close()

    if task_info is None:
        raise HTTPException(status_code=404, detail="Task not found")

    # If processing, return 202 BUT include the progress data
    if task_info["status"] != "completed" and task_info["status"] != "failed":
        return JSONResponse(
            status_code=202,
            content={
                "task_id": task_id,
                "status": task_info["status"],
                "progress": task_info.get("progress"),
                "result": None
            }
        )
    return {
        "task_id": task_id,
        "status": task_info["status"],
        "result": task_info.get("result")
    }

# @router.get("/status/{task_id}")
# async def get_status(task_id: str):
#     task_info = results_db.get(task_id)
#     if task_info is None:
#         raise HTTPException(status_code=404, detail="Task not found")
#     return {
#         "task_id": task_id,
#         "status": task_info["status"],
#         "result": task_info.get("result")  # may be None if not finished
#     }

@router.get("/download/{task_id}")
async def download_report(task_id: str):
    # Construct the path to where your PDF is actually stored on the server
    # If it's in a folder called 'output'
    file_path = f"output/RetailBoost_Japan_Retail_Strategy_Deck_with_Forecasts.pdf"
    
    if os.path.exists(file_path):
        return FileResponse(
            path=file_path, 
            filename="RetailBoost_Japan_Retail_Strategy_Deck_with_Forecasts.pdf", 
            media_type='application/pdf'
        )
    return {"error": "File not found on server"}