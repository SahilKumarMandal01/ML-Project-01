"""
Flask application for serving ML predictions.

This service:
1. Renders a landing page.
2. Accepts user input through an HTML form.
3. Uses the prediction pipeline to generate math score predictions.
4. Returns results back to the UI.

Integrated with:
- CustomException for structured error handling
- Centralized logging
"""

import sys
from flask import Flask, request, render_template

from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.exception import CustomException
from src.logger import logging


# -------------------------------------------------------------------------
# Flask Application Initialization
# -------------------------------------------------------------------------
application = Flask(__name__)
app = application


# -------------------------------------------------------------------------
# Landing Page Route
# -------------------------------------------------------------------------
@app.route("/")
def index():
    """
    Render the landing page.
    """
    logging.info("Rendering index page.")
    return render_template("index.html")


# -------------------------------------------------------------------------
# Prediction Route
# -------------------------------------------------------------------------
@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    """
    Handle prediction requests from the UI.

    GET  -> Render input form
    POST -> Run prediction pipeline and return result
    """
    if request.method == "GET":
        logging.info("Rendering prediction form.")
        return render_template("home.html")

    try:
        logging.info("Received prediction request.")

        # -----------------------------------------------------------------
        # Collect and validate form inputs
        # -----------------------------------------------------------------
        custom_data = CustomData(
            gender=request.form.get("gender"),
            race_ethnicity=request.form.get("ethnicity"),
            parental_level_of_education=request.form.get(
                "parental_level_of_education"
            ),
            lunch=request.form.get("lunch"),
            test_preparation_course=request.form.get("test_preparation_course"),
            reading_score=float(request.form.get("reading_score")),
            writing_score=float(request.form.get("writing_score")),
        )

        # Convert input to DataFrame
        input_df = custom_data.get_data_as_data_frame()
        logging.debug(f"Prediction input DataFrame:\n{input_df}")

        # -----------------------------------------------------------------
        # Run prediction pipeline
        # -----------------------------------------------------------------
        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(input_df)

        math_score = round(float(prediction[0]), 2)
        logging.info(f"Prediction successful. Math score: {math_score}")

        return render_template("home.html", results=math_score)

    except Exception as e:
        logging.error("Error occurred during prediction.", exc_info=True)
        raise CustomException(e, sys)


# -------------------------------------------------------------------------
# Application Entry Point
# -------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0")
