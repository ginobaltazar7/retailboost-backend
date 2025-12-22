import pandas as pd
import matplotlib
matplotlib.use('Agg') # Save memory
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from meteostat import Point, Daily
import io
import base64
import gc 

def fig_to_base64(fig):
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight', dpi=120)
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf-8')

def set_style():
    sns.set_style("white")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.linestyle'] = '--'

def generate_retail_insights(data_map):
    print("   -> Analytics: Starting Insight Generation...", flush=True)
    
    try:
        set_style()
        charts_payload = []
        growth_metric = 0.0 # Default value

        visit_data = data_map.get('air_visit_data')
        store_info = data_map.get('air_store_info')
        forecast_agg_df = data_map.get('forecast_data')
        date_info = data_map.get('date_info')

        if visit_data is None or store_info is None:
            return {"error": "Missing core data tables"}

        # Pre-processing
        visit_data['visit_date'] = pd.to_datetime(visit_data['visit_date'])
        visit_data['visitors'] = pd.to_numeric(visit_data['visitors'], downcast='unsigned')
        
        df = pd.merge(visit_data, store_info, on='air_store_id', how='left')

        del visit_data
        del store_info
        gc.collect()

        # ----------------------------------
        # GENERATE STATIC CHARTS (PDF Style)
        # ----------------------------------

        # Chart 1: Holidays
        if date_info is not None:
            date_info['visit_date'] = pd.to_datetime(date_info['calendar_date'])
            df_holiday = pd.merge(df[['visit_date', 'visitors']], date_info[['visit_date', 'holiday_flg']], on='visit_date', how='left')
            df_holiday['day_type'] = df_holiday['holiday_flg'].map({1: 'Holiday', 0: 'Working Day'})
            
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            sns.boxplot(x='day_type', y='visitors', data=df_holiday, width=0.5, ax=ax1)
            ax1.set_title('Holiday Impact', fontsize=14)
            charts_payload.append({"title": "Operations & Holidays", "image": fig_to_base64(fig1)})
            plt.close(fig1)
            del df_holiday

        # Chart 2: Top Genres
        top_genres = df.groupby('air_genre_name', observed=True)['visitors'].sum().sort_values(ascending=True).tail(10)
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.barh(top_genres.index, top_genres.values, color=sns.color_palette("viridis", 10))
        ax2.set_title('Top Genres', fontsize=14)
        charts_payload.append({"title": "Portfolio Mix", "image": fig_to_base64(fig2)})
        plt.close(fig2)

        # Chart 3: Forecast
        if forecast_agg_df is not None:
            if 'id' in forecast_agg_df.columns:
                forecast_agg_df[['air_store_id', 'visit_date']] = forecast_agg_df['id'].str.rsplit('_', n=1, expand=True)
                forecast_agg_df['visit_date'] = pd.to_datetime(forecast_agg_df['visit_date'])
            
            hist_daily = df.groupby('visit_date')['visitors'].sum().reset_index()
            fcast_daily = forecast_agg_df.groupby('visit_date')['visitors'].sum().reset_index()
            history_zoom = hist_daily.sort_values('visit_date').tail(90)

            # CALCULATE GROWTH METRIC HERE
            avg_hist = history_zoom['visitors'].mean()
            avg_fcast = fcast_daily['visitors'].mean()
            if avg_hist > 0:
                growth_metric = ((avg_fcast - avg_hist) / avg_hist) * 100

            fig6, ax6 = plt.subplots(figsize=(12, 6))
            ax6.plot(history_zoom['visit_date'], history_zoom['visitors'], color='black', label='History')
            ax6.plot(fcast_daily['visit_date'], fcast_daily['visitors'], color='blue', linestyle='--', label='Forecast')
            ax6.set_title(f'Demand Forecast (Projected {growth_metric:+.1f}% Growth)', fontsize=14)
            ax6.legend()
            charts_payload.append({"title": "Future Demand Forecast", "image": fig_to_base64(fig6)})
            plt.close(fig6)

        # ----------------------------------
        # GENERATE JSON (Interactive for React)
        # ----------------------------------
        
        # 1. Bar Data
        genre_payload = top_genres.sort_values(ascending=False).reset_index().rename(columns={'air_genre_name': 'name', 'visitors': 'value'}).to_dict(orient='records')

        # 2. Line Data
        hist_daily = df.groupby('visit_date')['visitors'].sum().reset_index()
        h_zoom = hist_daily.sort_values('visit_date').tail(90).copy()
        h_zoom['date'] = h_zoom['visit_date'].dt.strftime('%Y-%m-%d')
        h_zoom = h_zoom.rename(columns={'visitors': 'history_value'})
        hist_list = h_zoom[['date', 'history_value']].to_dict(orient='records')

        cast_list = []
        if forecast_agg_df is not None:
            f_json = fcast_daily.copy()
            f_json['date'] = f_json['visit_date'].dt.strftime('%Y-%m-%d')
            f_json = f_json.rename(columns={'visitors': 'forecast_value'})
            cast_list = f_json[['date', 'forecast_value']].to_dict(orient='records')

        combined_timeline = hist_list + cast_list

        del df
        gc.collect()

        return {
            "generated_charts": charts_payload,
            "genre_bar_chart": genre_payload,
            "demand_line_chart": combined_timeline,
            # NEW: Passing the calculated number to the frontend
            "metrics": {
                "growth_pct": round(growth_metric, 2)
            },
            "weather_summary": {"status": "skipped"}
        }

    except Exception as e:
        print(f"   -> Analytics Fatal Error: {e}", flush=True)
        return { "error": str(e), "generated_charts": [], "genre_bar_chart": [], "demand_line_chart": [] }
