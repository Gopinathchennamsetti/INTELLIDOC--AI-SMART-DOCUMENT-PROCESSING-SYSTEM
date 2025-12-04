import os
import logging
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from config import (
    UPLOAD_FOLDER,
    CLUSTERED_FOLDER,
    CONSOLIDATED_FOLDER,
    DUPLICATE_FOLDER,
    SUMMARY_FOLDER,
    SECRET_KEY,
)
from file_utils import read_docx
from clustering import cluster_documents
from consolidate import consolidate_document
from summary import summarize_document
from evaluation import compare_folders

app = Flask(__name__)
app.secret_key = SECRET_KEY

logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Home page – just renders the home template.
    """
    return render_template('home.html', clusters=None)


@app.route('/cluster', methods=['GET', 'POST'])
def cluster():
    """
    Handles clustering of documents.
    """
    try:
        return cluster_documents(request)
    except Exception as e:
        logging.error(f"Error in cluster route: {e}")
        return render_template('error.html', error_message="An error occurred while clustering documents."), 500


@app.route('/consolidate', methods=['GET', 'POST'])
def consolidate():
    """
    Handles consolidation of documents.
    """
    try:
        logging.info("Calling consolidate_document()")
        return consolidate_document(request)
    except Exception as e:
        logging.error(f"Error in consolidate route: {e}")
        return render_template('error.html', error_message="An error occurred while consolidating documents."), 500


@app.route('/summary', methods=['GET', 'POST'])
def summary():
    """
    Handles summarization of documents.
    """
    try:
        return summarize_document(request)
    except Exception as e:
        logging.error(f"Error in summary route: {e}")
        return render_template('error.html', error_message="An error occurred while summarizing documents."), 500


@app.route('/view_clusters', methods=['GET'])
def view_clusters():
    """
    Displays list of clustered docs.
    """
    try:
        files = [f for f in os.listdir(CLUSTERED_FOLDER) if f.endswith('.docx')]
        return render_template('view_clusters.html', files=files)
    except Exception as e:
        logging.error(f"Error in view_clusters route: {e}")
        return render_template('error.html', error_message="An error occurred while viewing clusters."), 500


@app.route('/view_consolidation', methods=['GET'])
def view_consolidation():
    """
    Displays list of consolidated docs.
    """
    try:
        files = [f for f in os.listdir(CONSOLIDATED_FOLDER) if f.endswith('.docx')]
        return render_template('view_consolidation.html', files=files)
    except Exception as e:
        logging.error(f"Error in view_consolidation route: {e}")
        return render_template('error.html', error_message="An error occurred while viewing consolidations."), 500


@app.route('/view_summary', methods=['GET'])
def view_summary():
    """
    Displays list of summary docs.
    """
    try:
        files = [f for f in os.listdir(SUMMARY_FOLDER) if f.endswith('.docx')]
        return render_template('view_summary.html', files=files)
    except Exception as e:
        logging.error(f"Error in view_summary route: {e}")
        return render_template('error.html', error_message="An error occurred while viewing summaries."), 500


@app.route('/cluster_file/<filename>')
def cluster_file(filename):
    """
    View a specific clustered file.
    """
    try:
        file_path = os.path.join(CLUSTERED_FOLDER, filename)
        if not os.path.isfile(file_path):
            return render_template('error.html', error_message="File not found."), 404
        text = read_docx(file_path)
        return render_template('view_file.html', content=text)
    except Exception as e:
        logging.error(f"Error in cluster_file route: {e}")
        return render_template('error.html', error_message="An error occurred while reading the cluster file."), 500


@app.route('/consolidated_file/<filename>')
def consolidated_file(filename):
    """
    View a specific consolidated file.
    """
    try:
        file_path = os.path.join(CONSOLIDATED_FOLDER, filename)
        if not os.path.isfile(file_path):
            return render_template('error.html', error_message="File not found."), 404
        text = read_docx(file_path)
        return render_template('view_file.html', content=text)
    except Exception as e:
        logging.error(f"Error in consolidated_file route: {e}")
        return render_template('error.html', error_message="An error occurred while reading the consolidated file."), 500


@app.route('/summary_file/<filename>')
def summary_file(filename):
    """
    View a specific summary file.
    """
    try:
        file_path = os.path.join(SUMMARY_FOLDER, filename)
        if not os.path.isfile(file_path):
            return render_template('error.html', error_message="File not found."), 404
        text = read_docx(file_path)
        return render_template('view_file.html', content=text)
    except Exception as e:
        logging.error(f"Error in summary_file route: {e}")
        return render_template('error.html', error_message="An error occurred while reading the summary file."), 500


@app.route('/download_clustered/<filename>')
def download_clustered(filename):
    """
    Download a clustered file.
    """
    try:
        file_path = os.path.join(CLUSTERED_FOLDER, filename)
        if not os.path.isfile(file_path):
            return render_template('error.html', error_message="File not found."), 404
        return send_from_directory(CLUSTERED_FOLDER, filename, as_attachment=True)
    except Exception as e:
        logging.error(f"Error in download_clustered route: {e}")
        return render_template('error.html', error_message="An error occurred while downloading the clustered file."), 500


@app.route('/download_consolidated/<filename>')
def download_consolidated(filename):
    """
    Download a consolidated file.
    """
    try:
        file_path = os.path.join(CONSOLIDATED_FOLDER, filename)
        if not os.path.isfile(file_path):
            return render_template('error.html', error_message="File not found."), 404
        return send_from_directory(CONSOLIDATED_FOLDER, filename, as_attachment=True)
    except Exception as e:
        logging.error(f"Error in download_consolidated route: {e}")
        return render_template('error.html', error_message="An error occurred while downloading the consolidated file."), 500


@app.route('/download_summary/<filename>')
def download_summary(filename):
    """
    Download a summary file.
    """
    try:
        file_path = os.path.join(SUMMARY_FOLDER, filename)
        if not os.path.isfile(file_path):
            return render_template('error.html', error_message="File not found."), 404
        return send_from_directory(SUMMARY_FOLDER, filename, as_attachment=True)
    except Exception as e:
        logging.error(f"Error in download_summary route: {e}")
        return render_template('error.html', error_message="An error occurred while downloading the summary file."), 500


@app.route('/evaluate')
def compare_folders_route():
    """
    Compares clustered vs consolidated folders and writes a CSV, then shows results.
    """
    try:
        output_csv = os.path.join(CONSOLIDATED_FOLDER, 'comparision_results.csv')
        compare_folders(CLUSTERED_FOLDER, CONSOLIDATED_FOLDER, output_csv)
        return redirect(url_for('show_results'))
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        return render_template('error.html', error_message=f"File not found: {e}"), 404
    except pd.errors.EmptyDataError:
        logging.error("The CSV file is empty or not found.")
        return render_template('error.html', error_message="The CSV file is empty or not found."), 400
    except Exception as e:
        logging.error(f"Unexpected error in evaluate route: {e}")
        return render_template('error.html', error_message=f"An unexpected error occurred: {e}"), 500


@app.route('/show_results', methods=['GET'])
def show_results():
    """
    Displays comparison results from the CSV file.
    """
    try:
        csv_file = os.path.join(CONSOLIDATED_FOLDER, 'comparision_results.csv')
        if not os.path.isfile(csv_file):
            return render_template('error.html', error_message="Results file not found."), 404

        df = pd.read_csv(csv_file)
        return render_template('results.html', tables=[df.to_html(classes='data')], titles=df.columns.values)
    except Exception as e:
        logging.error(f"Error in show_results route: {e}")
        return render_template('error.html', error_message=f"An error occurred while displaying the results: {e}"), 500


if __name__ == '__main__':
    app.run(debug=True)