import os
import logging
import pandas as pd
from bert_score import score as bert_score
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def read_docx(file_path):
    try:
        doc = Document(file_path)
        return '\n'.join([para.text for para in doc.paragraphs])
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")
        return ""


def compute_bertscore(ref_text, hyp_text):
    try:
        P, R, F1 = bert_score([hyp_text], [ref_text], lang='en', verbose=True)
        return P.mean().item(), R.mean().item(), F1.mean().item()
    except Exception as e:
        logging.error(f"Error computing BERTScore: {e}")
        return 0, 0, 0


def compute_cosine_similarity(ref_text, hyp_text):
    try:
        vectorizer = TfidfVectorizer().fit_transform([ref_text, hyp_text])
        vectors = vectorizer.toarray()
        cosine_sim = cosine_similarity(vectors)
        return cosine_sim[0, 1]
    except Exception as e:
        logging.error(f"Error computing cosine similarity: {e}")
        return 0


def compare_folders(ref_folder, hyp_folder, output_csv):
    results = []
    try:
        ref_files = sorted(os.listdir(ref_folder))
        hyp_files = sorted(os.listdir(hyp_folder))

        for ref_file, hyp_file in zip(ref_files, hyp_files):
            if ref_file.endswith('.docx') and hyp_file.endswith('.docx'):
                ref_path = os.path.join(ref_folder, ref_file)
                hyp_path = os.path.join(hyp_folder, hyp_file)

                reference_text = read_docx(ref_path)
                hypothesis_text = read_docx(hyp_path)

                P, R, F1 = compute_bertscore(reference_text, hypothesis_text)
                cosine_sim = compute_cosine_similarity(reference_text, hypothesis_text)

                result = {
                    'File': ref_file,
                    'BERTScore Precision': P,
                    'BERTScore Recall': R,
                    'BERTScore F1': F1,
                    'Cosine Similarity': cosine_sim
                }
                results.append(result)

        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        logging.info(f'Results saved to {output_csv}')
    except Exception as e:
        logging.error(f"Error comparing folders: {e}")