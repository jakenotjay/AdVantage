# AdVantage - Winner of the Earth-i Vantage coding challenge #2

Instructions for running the platform can be found in main.ipynb.

# Model
The model is built on top of the pretrained object detection model from Yolov5 with further training for aircraft/boats.

The aircraft model provides the best results with a recall > 90%.

The model was trained using data acquired from a multitude of sources, including Airbus, Vantage and Earth Engine and works with partial cloud cover.

# Vantage
Specific SDK for the Earth-i vantage platform can be found within vantage_api, allowing you to query for video using bounding boxes and airport IATA codes.

# AdVantage
The advantage folder contains the bulk of the work including pipelines, handlers etc.

For non-vantage platform video, remove VideoAttachGeoData from the pipeline, as the current Geographic processor is made specifically for XML from the Vantage platform.
