import asyncio
import uuid

# In a real app, replace this dict with some db connection like Postgres or Redis
results_db = {}

def _run_model_task_sync_wrapper(task_id, data_payload):
    """
    A synchronous wrapper function designed to be used with FastAPI's 
    BackgroundTasks utility, which runs within its own event loop context.
    """
    # This wrapper function needs an event loop to run the async function
    # It assumes that wherever it's called (e.g., in a FastAPI endpoint 
    # being called by TestClient) there is already an event loop available.
    
    # In a synchronous test environment using TestClient, FastAPI handles this.
    # We use asyncio.run in testing context only IF we run this file in isolation.
    # In production/testclient context, FastAPI manages the loop.
    try:
        loop = asyncio.get_running_loop()
        # If the loop is running, schedule the actual async task
        loop.create_task(run_model_task(task_id, data_payload))
    except RuntimeError:
        # If no loop is running (e.g., you try to call this manually in a script)
        # If no loop, run it directly
        asyncio.run(run_model_task(task_id, data_payload))
        # Handle gracefully or provide a fallback
        print(f"Warning: No running event loop to schedule task {task_id}")
        # Optionally run it synchronously if needed for specific tests where async isn't available
        # asyncio.run(run_model_task(task_id, data_payload))


def create_task_entry(data_payload: str, background_tasks=None):
    """
    Creates a new task ID and entry in the database.
    Accepts an optional background_tasks object (from FastAPI Depends/Request).
    """
    task_id = str(uuid.uuid4())
    results_db[task_id] = {"status": "processing", "result": None}
    
    if background_tasks:
        # If we are in a live FastAPI context (prod or TestClient), 
        # use FastAPI's mechanism to run the task in the background.
        background_tasks.add_task(run_model_task, task_id, data_payload)
        # Note: We pass the async function *directly* to add_task, 
        # FastAPI handles the event loop management for it.
    else:
        # Fallback for when we call this function manually in a simple script 
        # outside of a web request context (less common).
        _run_model_task_sync_wrapper(task_id, data_payload)
        
    return task_id

def get_task_result(task_id: str):
    """Retrieves a task result from the database."""
    return results_db.get(task_id)

async def run_model_task(task_id, data_payload):
    """Simulates a long-running ML model or processing task."""
    await asyncio.sleep(5) # Simulate processing delay
    results_db[task_id]["status"] = "completed"
    results_db[task_id]["result"] = {
        "message": f"Processed data: {data_payload}"
    }
