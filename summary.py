import os
import logging
from flask import render_template
from config import CLUSTERED_FOLDER, SUMMARY_FOLDER
from file_utils import read_docx, save_docx
from openai_utils import summarize_text


logging.basicConfig(
    filename='app.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def summarize_document(request):
    """
    Handles the summarization of selected document files.
    """
    if request.method == 'POST':
        selected_file = request.form.get('document')
        if selected_file:
            try:
                file_path = os.path.join(CLUSTERED_FOLDER, selected_file)
                if not os.path.isfile(file_path):
                    logging.error(f"File not found: {file_path}")
                    return render_template('error.html', error_message='Selected file does not exist.'), 404

                document_text = read_docx(file_path)

                try:
                    summary_text = summarize_text(document_text)
                except Exception as e:
                    logging.error(f"Gemini summarization error: {e}")
                    return render_template('error.html', error_message=f'Summarization error: {e}'), 500

                summary_path = os.path.join(SUMMARY_FOLDER, selected_file)
                save_docx(summary_path, summary_text)

                return render_template('success.html', message='Summary completed!')

            except FileNotFoundError:
                logging.error('The file was not found.')
                return render_template('error.html', error_message='The file was not found.'), 404
            except Exception as e:
                logging.error(f'Unexpected error: {e}')
                return render_template('error.html', error_message=f'An error occurred: {e}'), 500

    try:
        files = [f for f in os.listdir(CLUSTERED_FOLDER) if f.endswith('.docx')]
    except Exception as e:
        logging.error(f'Error listing files: {e}')
        return render_template('error.html', error_message=f'Error listing files: {e}'), 500

    return render_template('summary.html', files=files, message=None)