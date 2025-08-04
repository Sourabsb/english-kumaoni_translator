import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, pipeline, AutoTokenizer
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# --- 1. Load Models (this runs once when the app starts) ---
print("Loading all models...")

embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

base_model_name = "facebook/mbart-large-50-many-to-many-mmt"
# !!! IMPORTANT: Make sure this is your correct model ID
lora_adapter_id = "sourabsb/english_kumaoni-mbart_large" 

tokenizer = AutoTokenizer.from_pretrained(base_model_name, src_lang="en_XX", use_fast=False)
base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
finetuned_model = PeftModel.from_pretrained(base_model, lora_adapter_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
finetuned_model = finetuned_model.to(device)

print("✅ Models loaded.")

# --- 2. Load the Saved RAG Database ---
print("Loading the saved RAG database...")
vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={'k': 1, 'score_threshold': 0.95}
)
print("✅ RAG database loaded.")

# --- 3. Create the Translation Pipeline ---
translator_pipeline = pipeline(
    "translation",
    model=finetuned_model,
    tokenizer=tokenizer,
    src_lang="en_XX",
    tgt_lang="hi_IN",
    device=0
)

# --- 4. The Main Translation Function ---
def hybrid_translate(english_query):
    if not english_query:
        return "", "" # Return empty strings for both outputs
    
    retrieved_docs = retriever.get_relevant_documents(english_query)
    
    if retrieved_docs:
        source = "[Source: Database (Cache Hit)]"
        translation = retrieved_docs[0].metadata['kumaoni_translation']
    else:
        source = "[Source: Fine-tuned Model (Fallback)]"
        translation_result = translator_pipeline(english_query, max_length=50)
        translation = translation_result[0]['translation_text']
        
    return translation, source

# --- 5. Create the Custom Gradio UI using Blocks ---
with gr.Blocks(theme=gr.themes.Soft(), title="Kumaoni Translator") as demo:
    gr.Markdown("# Kumaoni AI Translator")
    gr.Markdown("A hybrid AI model for English to Kumaoni Roman translation, built by Sourab.")
    
    with gr.Row():
        with gr.Column():
            english_input = gr.Textbox(lines=4, placeholder="Type your English sentence here...", label="English Input")
            source_output = gr.Textbox(label="Source", interactive=False)
            
            with gr.Row():
                clear_button = gr.ClearButton()
                translate_button = gr.Button("Translate", variant="primary")
                
            # --- The gr.Examples block has been removed from here ---

        with gr.Column():
            kumaoni_output = gr.Textbox(lines=4, label="Kumaoni Translation", interactive=False)

    # Connect the button to the translation function
    translate_button.click(
        fn=hybrid_translate,
        inputs=english_input,
        outputs=[kumaoni_output, source_output]
    )
    
    # Connect the ClearButton to all relevant components
    clear_button.add([english_input, kumaoni_output, source_output])

# --- 6. Launch the App ---
demo.launch()