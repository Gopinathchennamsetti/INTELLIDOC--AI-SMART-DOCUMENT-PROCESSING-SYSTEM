
# import os
# from google import genai

# API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyC-ObFfAer6bCtZQ8h57HCXGboClx9JVYE")

# client = genai.Client(api_key=API_KEY)

# MODEL = "gemini-2.5-flash"  # or any other model from your list

# def quick_test():
#     try:
#         resp = client.models.generate_content(
#             model=MODEL,
#             contents="Reply with the single word: OK",
#         )
#         print("Model response:", resp.text)
#     except Exception as e:
#         print("Error calling model:", e)

# if __name__ == "__main__":
#     quick_test()






# test_gemini_functions.py
import os
from openai_utils import identify_duplicates, deduplicate_text, summarize_text

# Set your Gemini API key (or ensure it's in environment)
os.environ["GEMINI_API_KEY"] = "AIzaSyC-ObFfAer6bCtZQ8h57HCXGboClx9JVYE"

sample_text = """
Artificial Intelligence is transforming industries. AI is changing the way businesses operate.
Machine learning, a subset of AI, is widely used in data analysis. Artificial Intelligence is transforming industries.
Deep learning, another subset of AI, powers many modern applications. AI is changing the way businesses operate.
"""

print("=== Original Text ===")
print(sample_text)

print("\n=== Identify Duplicates ===")
duplicates = identify_duplicates(sample_text)
print(duplicates)

print("\n=== Deduplicated Text ===")
deduped = deduplicate_text(sample_text)
print(deduped)

print("\n=== Summary ===")
summary = summarize_text(sample_text)
print(summary)


# import os
# from google import genai

# # 1. Set your API key (or set GEMINI_API_KEY in your environment)
# API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyC-ObFfAer6bCtZQ8h57HCXGboClx9JVYE")

# client = genai.Client(api_key=API_KEY)

# def list_models():
#     try:
#         print("Listing available Gemini models for this API key:\n")
#         for model in client.models.list():
#             # Filter to only visible Gemini text/vision models
#             if model.name.startswith("models/gemini"):
#                 print(f"- {model.name}")
#     except Exception as e:
#         print("Error listing models:", e)

# if __name__ == "__main__":
#     list_models()