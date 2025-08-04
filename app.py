import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, pipeline, AutoTokenizer
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

print("Loading all models...")

embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

base_model_name = "facebook/mbart-large-50-many-to-many-mmt"
lora_adapter_id = "sourabsb/english_kumaoni-mbart_large" 

tokenizer = AutoTokenizer.from_pretrained(base_model_name, src_lang="en_XX", use_fast=False)
base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
finetuned_model = PeftModel.from_pretrained(base_model, lora_adapter_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
finetuned_model = finetuned_model.to(device)

print("✅ Models loaded.")

print("Loading the saved RAG database...")
vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={'k': 1, 'score_threshold': 0.95}
)
print("✅ RAG database loaded.")

translator_pipeline = pipeline(
    "translation",
    model=finetuned_model,
    tokenizer=tokenizer,
    src_lang="en_XX",
    tgt_lang="hi_IN",
    device=0
)

def hybrid_translate(english_query):
    if not english_query:
        return ""
    
    retrieved_docs = retriever.get_relevant_documents(english_query)
    
    if retrieved_docs:
        translation = retrieved_docs[0].metadata['kumaoni_translation']
    else:
        translation_result = translator_pipeline(english_query, max_length=50)
        translation = translation_result[0]['translation_text']
        
    return translation

with gr.Blocks(theme=gr.themes.Soft(), title="Kumaoni Translator") as demo:
    gr.Markdown("# English to Kumaoni Translator")
    gr.Markdown("A hybrid AI model for English to Kumaoni Roman translation, built by Sourab.")
    
    with gr.Row():
        with gr.Column():
            english_input = gr.Textbox(lines=5, placeholder="Type your English sentence here...", label="English Input")
            
            with gr.Row():
                clear_button = gr.ClearButton()
                translate_button = gr.Button("Translate", variant="primary")

        with gr.Column():
            kumaoni_output = gr.Textbox(lines=5, label="Kumaoni Translation", interactive=False)

    translate_button.click(
        fn=hybrid_translate,
        inputs=english_input,
        outputs=kumaoni_output
    )
    
    clear_button.add([english_input, kumaoni_output])

demo.launch()