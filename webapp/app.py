"""
Flask application for Langvio - process images and videos with natural language
"""

import os
import uuid
import time
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    send_from_directory,
)
from werkzeug.utils import secure_filename
import logging

# Import langvio
from langvio import create_pipeline
from langvio.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configure app
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev_key_change_in_production")

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "mp4", "mov", "avi", "webm"}

# Create required directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(os.path.join(app.root_path, "static", "results"), exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULTS_FOLDER"] = RESULTS_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100MB max-limit


# Initialize Langvio pipeline once
def create_langvio_pipeline():
    try:
        # Create config with good defaults
        config = Config()

        # Higher confidence for better visualization
        if "yoloe_large" in config.config["vision"]["models"]:
            config.config["vision"]["models"]["yoloe_large"]["confidence"] = 0.5
        elif "yolo" in config.config["vision"]["models"]:
            config.config["vision"]["models"]["yolo"]["confidence"] = 0.5

        # Set visualization options
        config.config["media"]["output_dir"] = app.config["RESULTS_FOLDER"]
        config.config["media"]["visualization"] = {
            "box_color": [0, 120, 255],  # Orange boxes
            "text_color": [255, 255, 255],  # White text
            "line_thickness": 2,  # Line thickness
            "show_attributes": True,  # Show attributes
            "show_confidence": False,  # Hide confidence to reduce clutter
        }

        # Try to use the best available model
        for model in ["yoloe_large", "yoloe", "yolo"]:
            if model in config.config["vision"]["models"]:
                pipeline = create_pipeline(vision_name=model)
                logger.info(f"Created pipeline with model: {model}")
                return pipeline

        # Fallback to default
        pipeline = create_pipeline()
        logger.info("Created pipeline with default model")
        return pipeline
    except Exception as e:
        logger.error(f"Error creating pipeline: {e}")
        return None


# Initialize pipeline
pipeline = create_langvio_pipeline()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def is_video(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {
        "mp4",
        "mov",
        "avi",
        "webp",
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process_media():
    if "file" not in request.files:
        flash("No file part")
        return redirect(request.url)

    file = request.files["file"]
    query = request.form.get("query", "Describe what is in this image/video")

    if file.filename == "":
        flash("No selected file")
        return redirect(request.url)

    if file and allowed_file(file.filename):
        # Generate a unique filename to avoid collisions
        original_filename = secure_filename(file.filename)
        filename = f"{int(time.time())}_{uuid.uuid4().hex}_{original_filename}"
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

        # Save the uploaded file
        file.save(file_path)

        try:
            # Check if pipeline is initialized
            if pipeline is None:
                flash(
                    "Error: Could not initialize the Langvio pipeline. Please check your installation."
                )
                return redirect(url_for("index"))

            # Process with Langvio
            logger.info(f"Processing file: {filename} with query: {query}")
            start_time = time.time()
            result = pipeline.process(query, file_path)
            processing_time = time.time() - start_time

            # Get the output path and copy to static directory for serving
            output_path = result.get("output_path", "")
            explanation = result.get("explanation", "No explanation provided.")

            if not output_path or not os.path.exists(output_path):
                flash("Error processing the file. No output was generated.")
                return redirect(url_for("index"))

            # Create a unique name for the result file
            output_filename = os.path.basename(output_path)
            destination = os.path.join(
                app.root_path, "static", "results", output_filename
            )

            # Copy the result file to the static directory
            import shutil

            shutil.copy2(output_path, destination)

            # Determine if the file is a video or image
            is_video_file = is_video(filename)
            result_url = url_for("static", filename=f"results/{output_filename}")

            # Get any statistics if available
            stats = {}
            if "detections" in result and "summary" in result["detections"]:
                stats = result["detections"]["summary"]

            # Get counts of detected objects
            object_counts = {}
            if "detections" in result and "0" in result["detections"]:
                from collections import Counter

                counts = Counter()
                for det in result["detections"]["0"]:
                    if "label" in det:
                        counts[det["label"]] += 1
                object_counts = dict(counts.most_common())

            # For videos, get additional stats
            video_stats = {}
            if is_video_file and "detections" in result:
                frame_count = len(
                    [k for k in result["detections"].keys() if k.isdigit()]
                )
                video_stats["frames_processed"] = frame_count

                # Track unique objects if available
                tracked_objects = set()
                for frame_key, detections in result["detections"].items():
                    if not frame_key.isdigit():
                        continue
                    for det in detections:
                        if "track_id" in det and "label" in det:
                            tracked_objects.add((det["label"], det["track_id"]))

                if tracked_objects:
                    from collections import Counter

                    track_counts = Counter([label for label, _ in tracked_objects])
                    video_stats["unique_objects"] = dict(track_counts.most_common())

            # Return the results page
            return render_template(
                "result.html",
                result_url=result_url,
                explanation=explanation,
                is_video=is_video_file,
                query=query,
                processing_time=round(processing_time, 2),
                stats=stats,
                object_counts=object_counts,
                video_stats=video_stats,
            )

        except Exception as e:
            import traceback

            logger.error(f"Error processing file: {str(e)}")
            logger.error(traceback.format_exc())
            flash(f"Error processing your file: {str(e)}")
            return redirect(url_for("index"))
    else:
        flash(
            "File type not allowed. Please upload an image (png, jpg, jpeg) or video (mp4, mov, avi, webm)."
        )
        return redirect(url_for("index"))


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/results/<filename>")
def result_file(filename):
    return send_from_directory(app.config["RESULTS_FOLDER"], filename)


if __name__ == "__main__":
    # Check if pipeline was created successfully
    if pipeline is None:
        logger.error(
            "Failed to initialize Langvio pipeline. Check if Langvio is installed correctly."
        )
    else:
        app.run(debug=True, host="0.0.0.0")
