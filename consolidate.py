import os
import logging
from flask import render_template, request
from config import CLUSTERED_FOLDER, DUPLICATE_FOLDER, CONSOLIDATED_FOLDER, SUMMARY_FOLDER
from file_utils import process_single_document


def consolidate_document(request):
    """
    Process a selected document for consolidation based on POST request.
    """
    if request.method == 'POST':
        selected_file = request.form.get('document')
        logging.info(f"Selected file for consolidation: {selected_file}")

        if selected_file:
            try:
                file_path = os.path.join(CLUSTERED_FOLDER, selected_file)

                if not os.path.exists(file_path):
                    logging.info(f'File does not exist: {file_path}')
                    return render_template('consolidate.html', files=get_files(),
                                           message="The selected file does not exist.")

                try:
                    process_single_document(file_path, DUPLICATE_FOLDER, CONSOLIDATED_FOLDER, SUMMARY_FOLDER)
                    logging.info(f'Consolidation completed: {file_path}')
                    return render_template('success.html', message='Document consolidation completed successfully!')

                except Exception as e:
                    logging.error(f'Error processing document: {e}')
                    return render_template('consolidate.html', files=get_files(),
                                           message="An error occurred while processing the document.")

            except Exception as e:
                logging.error(f'Error with file path or processing: {e}')
                return render_template('consolidate.html', files=get_files(),
                                       message="An error occurred. Please try again.")

    files = get_files()
    return render_template('consolidate.html', files=files, message=None)


def get_files():
    """
    Retrieve a list of .docx files from the clustered folder.
    """
    try:
        files = [f for f in os.listdir(CLUSTERED_FOLDER) if f.endswith('.docx')]
        logging.info(f'Files in {CLUSTERED_FOLDER}: {files}')
    except Exception as e:
        logging.error(f'Error listing files in {CLUSTERED_FOLDER}: {e}')
        files = []
    return files