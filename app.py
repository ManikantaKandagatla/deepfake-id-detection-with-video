from flask import Flask, request, jsonify, render_template, send_file
from core.detector import ImageDetector
import os
from werkzeug.utils import secure_filename
import csv
from datetime import datetime
import glob
import cv2
import numpy as np
import tempfile
import logging
from logging.handlers import RotatingFileHandler

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size for videos
app.config['RESULTS_FOLDER'] = 'results'

# Configure logging
if not os.path.exists('logs'):
    os.makedirs('logs')

file_handler = RotatingFileHandler('logs/app.log', maxBytes=10240, backupCount=10)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
file_handler.setLevel(logging.INFO)
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)
app.logger.info('Application startup')

# Ensure upload and results directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Initialize detector with token from environment
detector = ImageDetector()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return jsonify({'error': 'Invalid file type. Please upload an image.'}), 400
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        result = detector.analyze_image(filepath)
        
        # Clean up the uploaded file
        os.remove(filepath)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze-folder', methods=['POST'])
def analyze_folder():
    data = request.get_json()
    if not data or 'folder_path' not in data:
        return jsonify({'error': 'No folder path provided'}), 400
    
    folder_path = data['folder_path']
    if not os.path.isdir(folder_path):
        return jsonify({'error': 'Invalid folder path'}), 400
    
    # Create a unique filename for the results CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = f'analysis_results_{timestamp}.csv'
    csv_path = os.path.join(app.config['RESULTS_FOLDER'], csv_filename)
    
    try:
        # Get all image files in the folder
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))
            image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
        
        if not image_files:
            return jsonify({'error': 'No image files found in the specified folder'}), 400
        
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['file_name', 'is_real', 'confidence'])
            
            for image_path in image_files:
                try:
                    result = detector.analyze_image(image_path)
                    writer.writerow([
                        os.path.basename(image_path),
                        result['is_real'],
                        f"{result['confidence']:.2f}%"
                    ])
                except Exception as e:
                    print(f"Error processing {image_path}: {str(e)}")
                    continue
        
        return jsonify({
            'message': f'Analysis completed successfully. Processed {len(image_files)} images.',
            'csv_file': csv_filename
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze-video', methods=['POST'])
def analyze_video():
    if 'file' not in request.files:
        app.logger.error('No file uploaded in video analysis request')
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['file']
    if file.filename == '':
        app.logger.error('Empty filename in video analysis request')
        return jsonify({'error': 'No file selected'}), 400
        
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        app.logger.error(f'Invalid file type uploaded: {file.filename}')
        return jsonify({'error': 'Invalid file type. Please upload a video file (mp4, avi, or mov).'}), 400
    
    try:
        app.logger.info(f'Starting video analysis for file: {file.filename}')
        
        # Save the uploaded video temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        app.logger.info(f'Video saved temporarily at: {filepath}')
        
        # Open the video file
        cap = cv2.VideoCapture(filepath)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        
        app.logger.info(f'Video details - FPS: {fps}, Total Frames: {frame_count}, Duration: {duration:.2f} seconds')
        
        # Process frames at 10 fps
        frame_interval = int(fps / 10)
        if frame_interval < 1:
            frame_interval = 1
            
        frame_results = []
        current_frame = 0
        processed_frames = 0
        
        app.logger.info(f'Starting frame processing with interval: {frame_interval}')
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if current_frame % frame_interval == 0:
                processed_frames += 1
                app.logger.info(f'Processing frame {current_frame}/{frame_count} (Frame {processed_frames} of {frame_count//frame_interval})')
                
                # Save frame temporarily
                temp_frame = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
                cv2.imwrite(temp_frame.name, frame)
                
                # Analyze frame
                app.logger.info(f'Analyzing frame {current_frame}')
                result = detector.analyze_image(temp_frame.name)
                frame_results.append(result)
                app.logger.info(f'Frame {current_frame} analysis result: {result}')
                
                # Clean up temporary frame file
                os.unlink(temp_frame.name)
            
            current_frame += 1
        
        cap.release()
        
        # Clean up the uploaded video file
        os.remove(filepath)
        app.logger.info('Temporary video file removed')
        
        # Calculate final result
        fake_count = sum(1 for r in frame_results if not r['is_real'])
        total_frames = len(frame_results)
        
        # Calculate weighted scores based on confidence
        real_scores = []
        fake_scores = []
        
        for result in frame_results:
            if result['is_real']:
                # For real frames, higher confidence means more likely to be real
                real_scores.append(result['confidence'] / 100.0)
            else:
                # For fake frames, higher confidence means more likely to be fake
                fake_scores.append(result['confidence'] / 100.0)
        
        # Calculate average confidence for real and fake frames
        avg_real_confidence = sum(real_scores) / len(real_scores) if real_scores else 0
        avg_fake_confidence = sum(fake_scores) / len(fake_scores) if fake_scores else 0
        
        # Convert to percentages
        avg_real_confidence_percent = avg_real_confidence * 100
        avg_fake_confidence_percent = avg_fake_confidence * 100
        
        # Calculate weighted fake percentage
        fake_percentage = (len(fake_scores) / total_frames) * 100 if total_frames > 0 else 0
        
        # Determine if video is real based on average real confidence threshold
        is_video_real = avg_real_confidence_percent >= 80
        
        app.logger.info(f'Analysis complete - Total frames analyzed: {total_frames}, Fake frames: {len(fake_scores)}, '
                       f'Fake percentage: {fake_percentage:.2f}%, '
                       f'Average real confidence: {avg_real_confidence_percent:.2f}%, '
                       f'Average fake confidence: {avg_fake_confidence_percent:.2f}%, '
                       f'Video classified as: {"Real" if is_video_real else "Fake"}')
        
        final_result = {
            'is_real': is_video_real,
            'real_confidence': avg_real_confidence_percent,
            'fake_confidence': avg_fake_confidence_percent,
            'fake_frame_percentage': fake_percentage,
            'total_frames_analyzed': total_frames,
            'fake_frames': len(fake_scores)
        }
        
        app.logger.info(f'Final result: {final_result}')
        return jsonify(final_result)
        
    except Exception as e:
        app.logger.error(f'Error during video analysis: {str(e)}', exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/download-results/<filename>')
def download_results(filename):
    try:
        return send_file(
            os.path.join(app.config['RESULTS_FOLDER'], filename),
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True) 