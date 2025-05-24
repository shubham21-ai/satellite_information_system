import pandas as pd
import json
import os
from datetime import datetime

class SatelliteDataManager:
    def __init__(self):
        self.data_file = "satellite_data.json"
        self.load_data()

    def load_data(self):
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r') as f:
                self.data = json.load(f)
        else:
            self.data = {}

    def save_data(self):
        with open(self.data_file, 'w') as f:
            json.dump(self.data, f, indent=4)

    def append_satellite_data(self, satellite_name, data_type, data):
        if satellite_name not in self.data:
            self.data[satellite_name] = {}
        
        self.data[satellite_name][data_type] = {
            "data": data,
            "last_updated": datetime.now().isoformat()
        }
        self.save_data()

    def get_satellite_data(self, satellite_name, data_type=None):
        if satellite_name not in self.data:
            return None
        
        if data_type:
            return self.data[satellite_name].get(data_type)
        return self.data[satellite_name]

    def get_all_satellites(self):
        """Get a list of all satellites in the database"""
        return list(self.data.keys())

    def delete_satellite_data(self, satellite_name):
        """Delete all data for a specific satellite"""
        if satellite_name in self.data:
            del self.data[satellite_name]
            self.save_data()
            return True
        return False

    def get_dataframe(self, data_type=None):
        """Convert the data to a pandas DataFrame with serializable values"""
        rows = []
        for satellite, satellite_data in self.data.items():
            for dtype, info in satellite_data.items():
                if data_type and dtype != data_type:
                    continue
                
                # Convert dictionary data to string representation
                data = info["data"]
                if isinstance(data, dict):
                    data = json.dumps(data)
                
                rows.append({
                    "Satellite": satellite,
                    "Data Type": dtype,
                    "Value": data,
                    "Last Updated": info["last_updated"]
                })
        
        return pd.DataFrame(rows) 