import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

def log_to_sheets(spreadsheet_id, timestamp, object_class, confidence, cloud_url):
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name("config/credentials.json", scope)
        client = gspread.authorize(creds)
        sheet = client.open_by_key(spreadsheet_id).sheet1
        sheet.append_row([timestamp, object_class, confidence, cloud_url])
        print(f"Logged to Google Sheets: {timestamp}, {object_class}, {confidence}, {cloud_url}")
    except Exception as e:
        print(f"Error logging to Google Sheets: {str(e)}")