import gradio as gr
import mlflow
# from model import load_features
from data import preprocess_data
import json
import requests
import numpy as np
import pandas as pd
import os
from hydra import initialize_config_dir, compose

# current_dir = os.path.dirname(__file__)
# config_dir = os.path.join(current_dir, '..', 'configs')
# config_name="config"

# initialize_config_dir(config_dir)
# cfg = compose(config_name=config_name)

#---------------------------------------------------
# dont know what port should be used, maybe the same as for flask... 
# if needed, you can either take it from config (more generic way) or:
port_number = 5002
#---------------------------------------------------

project_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# define a parameter for each column in your raw dataset
def predict(date = None, open = None, high = None, low = None, close = None, adj_close = None, volume = None,
            symbol = None, security = None, gics_sector = None, gics_sub_inductry = None, News_All_News_Volume = None,
            News_Volume = None, News_Positive_Sentiment = None, News_Negative_Sentiment = None,
            News_New_Products = None, News_Layoffs = None, News_Analyst_Comments = None, News_Stocks = None,
            News_Dividends = None, News_Corporate_Earnings = None, News_Mergers_Acquisitions = None,
            News_Store_Openings = None, News_Product_Recalls = None, News_Adverse_Events = None,
            News_Personnel_Changes = None,News_Stock_Rumors = None
            ):


    # dict of column values -> pd.DataFrame
    features = {"Date": date, "Open": open, "High": high, "Low": low, "Close": close, "Adj Close": adj_close,
        "Volume": volume, "Symbol": symbol, "Security": security, "GICS Sector" : gics_sector, "GICS Sub-Industry": gics_sub_inductry,
        "News - All News Volume": News_All_News_Volume, "News - Volume": News_Volume, "News - Positive Sentiment": News_Positive_Sentiment,
        "News - Negative Sentiment": News_Negative_Sentiment, "News - New Products": News_New_Products,
        "News - Layoffs": News_Layoffs, "News - Analyst Comments": News_Analyst_Comments, "News - Stocks": News_Stocks,
        "News - Dividends": News_Dividends, "News - Corporate Earnings": News_Corporate_Earnings,
        "News - Mergers & Acquisitions": News_Mergers_Acquisitions, "News - Store Openings": News_Store_Openings,
        "News - Product Recalls": News_Product_Recalls, "News - Adverse Events": News_Adverse_Events,
        "News - Personnel Changes": News_Personnel_Changes, "News - Stock Rumors": News_Stock_Rumors
    }
    raw_df = pd.DataFrame(features, index=[0])  # only 1 example
    
    # Only transform the input data (no fit here)
    X, _ = preprocess_data(df = raw_df)
    
    # convert it into JSON
    example = X.iloc[0,:]
    example = json.dumps( 
        { "inputs": example.to_dict() }
    )

    payload = example

    # Send POST request with the payload to the deployed Model API
    # Here you can pass the port number at runtime using Hydra or using port_number
    response = requests.post(
        url=f"http://localhost:{port_number}/predict",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    
    # Change this to some meaningful output for your model
    # For regression, it returns the predicted value
    return response.json()

demo = gr.Interface(
    fn=predict,
    inputs = [
        gr.Text(label="Date"),
        gr.Number(label="Open"),
        gr.Number(label="High"),
        gr.Number(label="Low"),
        gr.Number(label="Close"),
        gr.Number(label="Adj Close"),
        gr.Number(label="Volume"),
        gr.Text(label="Symbol"),
        gr.Text(label="Security"),
        gr.Dropdown(label="GICS Sector", choices=['Communication Services', 'Health Care', 'Information Technology',
                                                    'Industrials', 'Utilities', 'Consumer Discretionary',
                                                    'Consumer Staples', 'Financials', 'Real Estate', 'Materials',
                                                    'Energy']),  
        gr.Text(label="GICS Sub-Industry"),
        gr.Number(label="News - All News Volume"),
        gr.Number(label="News - Volume"),
        gr.Number(label="News - Positive Sentiment"),
        gr.Number(label="News - Negative Sentiment"),
        gr.Number(label="News - New Products"),
        gr.Number(label="News - Layoffs"),
        gr.Number(label="News - Analyst Comments"),
        gr.Number(label="News - Stocks"),
        gr.Number(label="News - Dividends"),
        gr.Number(label="News - Corporate Earnings"),
        gr.Number(label="News - Mergers & Acquisitions"),
        gr.Number(label="News - Store Openings"),
        gr.Number(label="News - Product Recalls"),
        gr.Number(label="News - Adverse Events"),
        gr.Number(label="News - Personnel Changes"),
        gr.Number(label="News - Stock Rumors")
    ],
    
    # The outputs here will get the returned value from `predict` function
    # Don't know if should be converted to gr.Number
    outputs = gr.Text(label="prediction result"),
    examples=os.path.join(project_root_dir, 'gradio_examples', 'data', 'examples')
)

# Launch the web UI locally on port 5155
demo.launch(server_port = 5155)