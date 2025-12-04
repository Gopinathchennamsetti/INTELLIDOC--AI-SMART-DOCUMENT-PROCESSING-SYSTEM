
# AI-Powered Document De-Duplication and Consolidation for Efficient Information Management

## Project Overview

Our project aims to revolutionize information management with advanced AI strategies. We are developing an automated system that utilizes AI, particularly machine learning algorithms, to merge duplicate documents in a vast database. This approach will save time, enhance data quality, and reduce storage costs. Our tool ensures easy access to relevant data, promoting effective decision-making in organizations. The AI-Powered Document De-Duplication and Consolidation tool is designed to deliver precise and impactful results, potentially transforming how firms manage digital resources.




## Key Features

- **Document Upload and Management**: Effortlessly upload and organize various document formats through the web interface.
- **Clustering and Consolidation**: Clusters similar documents using Affinity Propagation and consolidates them with OpenAI’s GPT-3.5 Turbo for high-quality results.
- **Advanced Text Processing**: Preprocesses text with NLP techniques like stop word removal, lemmatization, and tokenization, and extracts features using TF-IDF.
- **Named Entity Recognition (NER)**: Extracts company names from document clusters for enhanced analysis.
- **Interactive Visualization**: Uses Plotly for dynamic clustering result visualizations, aiding in data exploration.
- **BERTScore Evaluation**: Measures semantic similarity between documents for detailed quality assessment.
- **Search Functionality**: Allows efficient searching within consolidated documents to quickly find relevant information.
- **User-Friendly Interface**: Provides a simple and intuitive UI for uploading, clustering, consolidating, and reviewing documents.
- **Efficient Scaling**: Handles both small and large datasets effectively, suitable for diverse document processing needs.




## Directory Structure

    project_root/
    │
    ├── main.py
    ├── routes.py
    ├── clustering.py
    ├── consolidate.py
    ├── summary.py
    ├── config.py
    ├── requirements.txt
    ├── file_utils.py
    ├── openai_utils.py
    ├── text_processing.py
    ├── evaluation.py
    ├── templates/
    │   ├── home.html
    │   ├── index.html
    │   ├── consolidate.html
    │   ├── summary.html
    │   ├── view_clusters.html
    │   ├── view_consolidation.html
    │   ├── view_file.html
    │   └── view_summary.html
    |
    └── README.txt
Here’s the complete file description with a detailed overview of each file and its purpose:

---

## File Descriptions

- **`main.py`**  
  Main Flask application file that orchestrates routes and integrates functionalities from other modules.

- **`clustering.py`**  
  Manages the vectorization and clustering of documents.  
  **Functions:**
  - `cluster_documents(documents)`: Clusters the provided documents based on their vectorized representations.

- **`consolidate.py`**  
  Handles the consolidation of clustered documents by removing duplicates.  
  **Functions:**
  - `consolidate_document(documents)`: Consolidates the documents in each cluster, removing duplicates and generating consolidated output.

- **`summary.py`**  
  Generates summaries for documents using GPT.  
  **Functions:**
  - `summarize_document(document)`: Produces a summary for the given document text using the GPT API.

- **`config.py`**  
  Configuration file for constants and settings.  
  **Variables:**
  - `OPENAI_API_KEY`: API key for OpenAI services.
  - `UPLOAD_FOLDER`: Directory for uploaded files.
  - `CLUSTERED_FOLDER`: Directory for storing clustered documents.
  - `CONSOLIDATED_FOLDER`: Directory for storing consolidated documents.
  - `DUPLICATE_FOLDER`: Directory for storing duplicate documents.
  - `SUMMARY_FOLDER`: Directory for storing summarized documents.
  - `SECRET_KEY`: Secret key for Flask session management.

- **`file_utils.py`**  
  Provides utility functions for file handling.  
  **Functions:**
  - `load_documents_from_filepaths(filepaths)`: Loads documents from specified file paths.
  - `process_single_document(file_path, duplicate_folder, consolidated_folder, summary_folder)`:The  function handles a document by checking for duplicates, consolidating and deduplicating its content, and generating a summary. It saves duplicates to the specified folder, consolidates the cleaned document, and stores the summary in their respective folders.
  - `read_docx(file_path)`: Reads text from a DOCX file.
  - `save_docx(file_path, text)`: Saves text to a DOCX file.

- **`openai_utils.py`**  
  Handles interactions with OpenAI API for document processing.  
  **Functions:**
  - `split_text_into_chunks(text,length)`The function divides a long text into smaller, manageable chunks to fit within token limits. It ensures that each chunk contains complete sentences and fits within the specified maximum chunk size.
  - `identify_duplicates(docs)`: Identifies and marks duplicates in the given documents.
  - `deduplicate_text(text)`: Removes duplicate text from the provided content.
  - `summarize_text(text)`: Summarizes the provided text using OpenAI's GPT.

- **`text_processing.py`**  
  Provides text processing utilities.  
  **Functions:**
  - `preprocess_text(text)`: Preprocesses the input text by cleaning and lemmatizing it.
  - `identify_company_name(texts)`: Identifies the most common company name from a list of texts.

- **`routes.py`**  
  Defines the routes for the Flask application, handling various functionalities related to document clustering, consolidation, summarization, and evaluation.  
  **Routes:**
  - **`/`**: Home page.
  - **`/cluster`**: Clusters documents.
  - **`/consolidate`**: Consolidates documents.
  - **`/summary`**: Summarizes documents.
  - **`/view_clusters`**: Lists clustered files.
  - **`/view_consolidation`**: Lists consolidated files.
  - **`/view_summary`**: Lists summary files.
  - **`/cluster_file/<filename>`**: Displays a clustered file.
  - **`/consolidated_file/<filename>`**: Displays a consolidated file.
  - **`/summary_file/<filename>`**: Displays a summary file.
  - **`/download_clustered/<filename>`**: Downloads a clustered file.
  - **`/download_consolidated/<filename>`**: Downloads a consolidated file.
  - **`/download_summary/<filename>`**: Downloads a summary file.
  - **`/evaluate`**: Compares folders and generates a CSV report.
  - **`/show_results`**: Shows comparison results from the CSV file.

- **`evaluation.py`**  
  Provides functions for evaluating and comparing documents.  
  **Functions:**
  - `compare_folders(clustered_folder, consolidated_folder, output_csv)`: Compares documents in the specified folders and saves the results to a CSV file.

- **`templates/`**  
  Contains HTML templates for the web interface.  
  **Templates:**
  - **`home.html`**: Home page template with options to cluster, consolidate, and summarize documents.
  - **`index.html`**: Page for initiating document clustering and displaying clustering results.
  - **`consolidate.html`**: Page for consolidating clustered documents and displaying consolidation results.
  - **`summary.html`**: Page for summarizing documents and displaying summary results.
  - **`view_clusters.html`**: Lists and provides access to clustered documents.
  - **`view_consolidation.html`**: Lists and provides access to consolidated documents.
  - **`view_summary.html`**: Lists and provides access to summary files.
  - **`view_file.html`**: Displays the content of a selected file (clustered, consolidated, or summary).
  - **`error.html`**: Error page template for displaying error messages.
  - **`success.html`**: Success page template for displaying completion messages.
  - **`cluster_viz.html`**:The  template displays interactive visualizations of document clusters using Plotly. It features a Plotly chart that visualizes cluster distributions and a link to download the visualization.
  - **`results.html`**:The  template displays the evaluation results from a CSV file as an HTML table. It includes options to view detailed data and download the results.
  - **`upload.html`**:The template provides a form for users to upload document files. It includes a file input field and a submit button to start the upload process.
---

## Requirements

To set up the environment for this project, make sure you have the following Python packages installed. You can use `pip` to install them from the `requirements.txt` file.

**Dependencies:**

- **Flask**: A lightweight WSGI web application framework.
- **numpy**: A library for numerical computations.
- **pandas**: A library for data manipulation and analysis.
- **spacy**: A library for natural language processing.
- **openai**: OpenAI's API client library for accessing GPT models.
- **nltk**: The Natural Language Toolkit for text processing.
- **python-docx**: A library for creating and updating Microsoft Word (.docx) files.
- **plotly**: A library for creating interactive plots and visualizations.
- **bert-score**: A library for computing BERT-based similarity scores.
- **scikit-learn**: A machine learning library for Python.
- **Werkzeug**: A comprehensive WSGI web application library.

**`requirements.txt`:**

```
Flask==2.3.0
numpy==1.24.2
pandas==2.0.1
spacy==3.5.0
openai==0.27.0
nltk==3.8.1
python-docx==0.8.11
plotly==5.15.0
bert-score==0.3.13
scikit-learn==1.2.2
Werkzeug==2.3.2
```

To install all dependencies, run:

```
pip install -r requirements.txt
```

---


## Setup Instructions

1. **Create a Virtual Environment**
   ```bash
   python -m venv .venv
   ```

2. **Activate the Virtual Environment**
   ```bash
   .venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up OpenAI API Key**
   Open the `config.py` file and update the `OPENAI_API_KEY` variable with your API key value:
   ```python
   OPENAI_API_KEY = 'your-api-key'
   ```

5. **Initialize NLP Models**
   ```python
   import spacy
   import nltk

   spacy.cli.download("en_core_web_sm")
   nltk.download('stopwords')
   nltk.download('punkt')
   ```


## App Usage

1. **Start the Flask Application**
   ```bash
   python app.py
   ```

2. **Access the Web Interface**
   Open your web browser and navigate to `http://127.0.0.1:5000`.

3. **Upload Documents**
   Use the provided interface to upload documents for processing.

4. **Initiate Clustering and Consolidation**
   Follow the UI prompts to cluster and consolidate documents.

5. **View Results**
   Check clustering results, consolidated documents, and interactive visualizations on the web interface.


