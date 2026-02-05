# Ejecta Detection

This project leverages deep learning to identify lunar ejecta (fresh impact craters with ray systems) from LROC imagery. It utilizes a dual-model approach analyzing both standard and color inverted imagery to maximize detection accuracy across varying lunar terrains.

## Features

- Dual Detection: Uses two specialized YOLOv8 models (n.pt for normal visual, inv.pt for structural/inverted visual) to reduce false negatives.
- Live Lunar Scanner: Connects directly to the USGS LROC WMS server to scan specific lunar coordinates in real-time.
- Web Dashboard: A full UI built with Streamlit for easy interaction, featuring a map gallery and downloadable CSV reports.
- Test-Time Augmentation (TTA): Automatically validates detections by rotating and flipping imagery during the scan to filter out noise.

##  Folder Structure

```
ejecta/
│── app.py                 # Main GUI application
│── requirements.txt       # Dependencies
│── ejecta_local.py        # Script for running on local files
|── scraper.py             # Script for scraping the LROC database
│── README.md              # Documentation
│── n.pt                   # normal model
|── inv.pt                 # inverted model
│── output                 # input folder for the local script
|── input                  # input folder for the local script
```
## Installation

### 1. Clone the repository
```bash
git clone https://github.com/Vysh-131/ejecta.git
```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### Model weights

Feel Free to contact me at vyshakhsanil13@gmail.com for the weights required for this project

(Make sure the weights are in the same root directory)

## Usage
```bash
streamlit run app.py
```

                                                                                                                                                                          Project by VYSH131






