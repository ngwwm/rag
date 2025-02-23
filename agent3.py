import os
import pyarrow as pa
import lancedb
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
from docx import Document
import PyPDF2
import pandas

# Directory for documents and model path
DOCUMENTS_DIR = "z:/"
MODEL_PATH = "C:/LLMs/lmstudio-community/DeepSeek-R1-Distill-Llama-8B-GGUF/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"

# Custom embedder using SentenceTransformers
class LocalEmbedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed(self, texts):
        return self.model.encode(texts, convert_to_numpy=True)

# Local LLM using llama-cpp-python
class LocalLLM:
    def __init__(self, model_path=MODEL_PATH):
        # Load the GGUF model
        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,  # Context length; adjust based on RAM (higher = more memory)
            #n_gpu_layers=0 if not Llama.nvidia_available() else -1,  # -1 uses GPU if available
            verbose=True  # Set to False to reduce logging
        )
    
    def generate(self, prompt, max_tokens=200):
        # Generate response
        response = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=0.7,  # Controls creativity; lower = more deterministic
            top_p=0.9  # Nucleus sampling; adjust for diversity
        )
        return response["choices"][0]["text"].strip()

# Function to extract text from .docx
def extract_text_from_docx(file_path):
    try:
        doc = Document(file_path)
        text = " ".join([para.text for para in doc.paragraphs if para.text])
        return text
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Function to extract text from .pdf
def extract_text_from_pdf(file_path):
    try:
        with open(file_path, 'rb') as file:  # Open in binary mode
            reader = PyPDF2.PdfReader(file)
            text = " ".join([reader.pages[i].extract_text() for i in range(len(reader.pages))])
            return text
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Function to extract text from .txt
def extract_text_from_txt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return text
    except UnicodeDecodeError:
        # Fallback to latin-1 if UTF-8 fails
        try:
            with open(file_path, 'r', encoding='latin-1') as file:
                text = file.read()
            print(f"Warning: {file_path} decoded with latin-1 due to UTF-8 failure")
            return text
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def create_document_agent():
    if not os.path.exists(DOCUMENTS_DIR):
        raise ValueError(f"Directory {DOCUMENTS_DIR} does not exist")

    # Initialize embedder and LLM
    embedder = LocalEmbedder()
    print("After LocalEmbedder...")
    llm = LocalLLM()
    print("After LocalLLM...")

    # Connect to LanceDB
    db = lancedb.connect("tmp/lancedb")

    # Define the schema using pyarrow
    schema = pa.schema([
        pa.field("vector", pa.list_(pa.float32(), 384)),  # Fixed-size vector of 384 floats
        pa.field("text", pa.string())                    # String type for text
    ])

    table = db.create_table(
        "documents",
        schema=schema,
        #schema={"vector": "vector(384)", "text": "string"},  # 384 matches all-MiniLM-L6-v2
        mode="overwrite"  # Overwrites existing table; use "create" to append
    )

    data = []
    success_count = 0
    fail_count = 0

    # Load documents into LanceDB
    print("Loading documents...")
    for file_path in os.listdir(DOCUMENTS_DIR):
        file_path = os.path.join(DOCUMENTS_DIR, file_path)

        if file_path.endswith(".docx"):
            text = extract_text_from_docx(file_path)
        elif file_path.endswith(".pdf"):
            text = extract_text_from_pdf(file_path)
        elif file_path.endswith(".txt"):
            text = extract_text_from_txt(file_path)
        else:
            print(f"Unsupported file type: {file_path}")
            fail_count += 1
            continue
    
        if text:
            # Generate embedding
            embedding = embedder.model.encode(text).tolist()  # Convert to list for LanceDB
            data.append({"vector": embedding, "text": text})
            print(f"Text extracted {file_path}!!!!!")
            success_count += 1
        else:
            print(f"Skipping {file_path} due to extraction failure")
            fail_count += 1

    print("Documents loaded successfully")
    # Print success and failure counts
    print(f"\nProcessing Summary:")
    print(f"Successful documents: {success_count}")
    print(f"Failed documents: {fail_count}")

    return embedder, llm, table

def main():
    try:
        # Create the agent components
        embedder, llm, table = create_document_agent()

        # Summarize all documents
        all_texts = [row["text"] for row in table.to_pandas().to_dict("records")]
        summary_prompt = (
            "You are an AI assistant tasked with summarizing documents. "
            "Summarize the following content:\n" + "\n".join(all_texts[:5000])  # Truncate to fit context
        )
        print("\nGenerating summary:")
        print(llm.generate(summary_prompt))

        # Interactive question loop
        while True:
            question = input("\nAsk a question about the documents (or 'quit' to exit): ")
            if question.lower() == 'quit':
                break
            query_vector = embedder.embed([question])[0]
            results = table.search(query_vector).limit(1).to_pandas()
            context = results["text"].iloc[0] if not results.empty else "No relevant information found in the documents."
            prompt = (
                "You are an AI assistant. Based on the following context from documents:\n"
                f"'{context}'\nAnswer this question: {question}"
            )
            print(llm.generate(prompt))

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
