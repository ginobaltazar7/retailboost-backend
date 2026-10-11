## RetailBoost Forecast Demo (Backend)

A FastAPI service that runs async demand-forecasting pipelines over retail data, with a Supabase Postgres-backed task status tracking. 

The pipeline, which runs on Render, forecasts demand using LightGBM with 5-fold cross-validation and stacking, temporally weighted toward recent signals, and generates a PDF strategy deck. 

Exposes /ingest, /results/{task_id}, and /download/{task_id} for the [companion frontend](https://github.com/ginobaltazar7/retailboost-frontend/).
