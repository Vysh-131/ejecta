import os
import time
import requests
import io
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO

try:
    from ejecta_local import EjectaDetector
except ImportError:
    print("ejecta_engine.py not found. Using internal simplified engine.")
    # Placeholder
    EjectaDetector = None 

# --- CONFIGURATION ---
class ScraperConfig:
    # WMS Server Settings (The Source)
    WMS_URL = "https://planetarymaps.usgs.gov/cgi-bin/mapserv"
    MAP_FILE = "/maps/earth/moon_simp_cyl.map"
    LAYER_NAME = "LROC_WAC"
    
    # Scanning Parameters
    SUPER_TILE_SIZE_DEG = 1.0  # Download 1.0x1.0 degree chunks (~30x30km)
    DOWNLOAD_RES = 3000        # Pixel resolution for the chunk (High Detail)
    
    # Output
    BASE_OUTPUT_DIR = "static/scans" # 'static' is standard for Web Apps

class LunarScraper:
    def __init__(self):
        print("Initializing LROC Scraper...")
        # Initialize the detection engine (Load models once)
        if EjectaDetector:
            self.detector = EjectaDetector()
        else:
            # Fallback for testing without the module
            self.detector = None
            print("No Detector Loaded. Scraper will only download maps.")

    def _fetch_super_tile(self, lat, lon):
        """
        Downloads a massive High-Res chunk from USGS.
        """
        min_lat, max_lat = lat, lat + ScraperConfig.SUPER_TILE_SIZE_DEG
        min_lon, max_lon = lon, lon + ScraperConfig.SUPER_TILE_SIZE_DEG
        
        params = {
            'map': ScraperConfig.MAP_FILE, 'SERVICE': 'WMS', 'VERSION': '1.1.1',
            'REQUEST': 'GetMap', 'LAYERS': ScraperConfig.LAYER_NAME, 'STYLES': '',
            'SRS': 'EPSG:4326',
            'BBOX': f"{min_lon},{min_lat},{max_lon},{max_lat}",
            'WIDTH': ScraperConfig.DOWNLOAD_RES, 'HEIGHT': ScraperConfig.DOWNLOAD_RES,
            'FORMAT': 'image/jpeg'
        }
        try:
            r = requests.get(ScraperConfig.WMS_URL, params=params, timeout=45)
            if r.status_code == 200:
                img_array = np.array(Image.open(io.BytesIO(r.content)))
                # Convert RGB (PIL) to BGR (OpenCV)
                return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Connection Error at {lat},{lon}: {e}")
        return None

    def _generate_quickmap_link(self, lat, lon, zoom=12):
        """
        Creates a clickable link to view the detected crater on the official site.
        """
        # LROC Quickmap URL format:
        # https://quickmap.lroc.asu.edu/layers?extent={lon_min},{lat_min},{lon_max},{lat_max}&proj=10
        # But a simpler point view is:
        return f"https://quickmap.lroc.asu.edu/layers?proj=10&center={lon},{lat}&zoom={zoom}"

    def _pixel_to_geo(self, px_x, px_y, tile_lat_start, tile_lon_start):
        """
        Converts pixel coordinates inside the Super Tile back to Global Lat/Lon.
        """
        deg_per_pixel = ScraperConfig.SUPER_TILE_SIZE_DEG / ScraperConfig.DOWNLOAD_RES
        
        # Calculate offset
        lat_offset = px_y * deg_per_pixel
        lon_offset = px_x * deg_per_pixel
        
        # Note: Image Y is top-down, Map Lat is bottom-up (usually)
        # But in WMS BBOX, we requested min_lat to max_lat.
        # Standard WMS images draw Max Lat at Y=0.
        real_lat = (tile_lat_start + ScraperConfig.SUPER_TILE_SIZE_DEG) - lat_offset
        real_lon = tile_lon_start + lon_offset
        
        return round(real_lat, 5), round(real_lon, 5)

    def run_scan_job(self, lat_min, lat_max, lon_min, lon_max, job_id="scan_001"):
        """
        🚀 WEBAPP ENTRY POINT
        This function is a Generator. It yields status updates so the UI 
        can show a progress bar (e.g., "Scanning tile 1 of 50...").
        """
        # Setup Output Folders
        job_dir = os.path.join(ScraperConfig.BASE_OUTPUT_DIR, job_id)
        img_dir = os.path.join(job_dir, "detections")
        if not os.path.exists(img_dir): os.makedirs(img_dir)
        
        # Create Grid
        lats = np.arange(lat_min, lat_max, ScraperConfig.SUPER_TILE_SIZE_DEG)
        lons = np.arange(lon_min, lon_max, ScraperConfig.SUPER_TILE_SIZE_DEG)
        
        total_tiles = len(lats) * len(lons)
        processed_count = 0
        all_detections = []

        yield {"status": "started", "total_tiles": total_tiles}

        for lat in lats:
            for lon in lons:
                processed_count += 1
                yield {"status": "scanning", "current": processed_count, "total": total_tiles, "lat": lat, "lon": lon}
                
                # 1. Download Super Tile
                super_tile = self._fetch_super_tile(lat, lon)
                if super_tile is None: continue
                
                # 2. Run Robust Detection (Slicing + TTA)
                if self.detector:
                    # We pass the huge image to the engine we wrote before
                    # It returns the image with boxes drawn, and a JSON list of hits
                    annotated_img, hits = self.detector.analyze_single_image(super_tile)
                else:
                    hits = [] # For testing without engine
                
                # 3. Process Hits
                if hits:
                    # Save the "Overview Map" for this tile
                    map_filename = f"map_{lat}_{lon}.jpg"
                    cv2.imwrite(os.path.join(img_dir, map_filename), annotated_img)
                    
                    for i, hit in enumerate(hits):
                        # Convert bbox to Real Coordinates
                        x1, y1, x2, y2 = hit['bbox']
                        cx, cy = (x1+x2)/2, (y1+y2)/2
                        
                        real_lat, real_lon = self._pixel_to_geo(cx, cy, lat, lon)
                        link = self._generate_quickmap_link(real_lat, real_lon)
                        
                        # Save a "Chip" (Zoomed in cutout) for the gallery
                        # Padding of 50px
                        h, w = super_tile.shape[:2]
                        cy1, cy2 = max(0, int(y1)-50), min(h, int(y2)+50)
                        cx1, cx2 = max(0, int(x1)-50), min(w, int(x2)+50)
                        chip = super_tile[cy1:cy2, cx1:cx2]
                        
                        chip_name = f"ejecta_{real_lat}_{real_lon}.jpg"
                        cv2.imwrite(os.path.join(img_dir, chip_name), chip)
                        
                        all_detections.append({
                            "Latitude": real_lat,
                            "Longitude": real_lon,
                            "Confidence": hit['confidence'],
                            "Model": hit['model'],
                            "LROC_Link": link,
                            "Image_Chip": chip_name,
                            "Source_Map": map_filename
                        })
                        
        # 4. Save Final Report
        csv_path = os.path.join(job_dir, "results.csv")
        if all_detections:
            df = pd.DataFrame(all_detections)
            df.to_csv(csv_path, index=False)
            yield {"status": "completed", "hits": len(all_detections), "report": csv_path}
        else:
            yield {"status": "completed", "hits": 0, "report": None}

# --- STANDALONE TESTER ---
if __name__ == "__main__":
    scraper = LunarScraper()
    
    # Test a small region (e.g., Copernicus)
    print("🧪 Starting Test Scan...")
    scanner = scraper.run_scan_job(9.0, 10.0, -21.0, -20.0, job_id="test_run")
    
    for update in scanner:
        print(f"   update: {update}")