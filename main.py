import streamlit as st
from PyPDF2 import PdfReader
import os
from google import genai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="Qubave AI - PDF Analysis",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Qubave AI branding
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .qubave-logo {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(45deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .chat-container {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .user-message {
        background: #667eea;
        color: white;
        padding: 1rem;
        border-radius: 15px 15px 5px 15px;
        margin: 1rem 0;
        max-width: 80%;
    }
    .ai-message {
        background: white;
        color: #333;
        padding: 1.5rem;
        border-radius: 15px 15px 15px 5px;
        margin: 1rem 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        border-left: 3px solid #667eea;
    }
    .footer-brand {
        text-align: center;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 10px;
        margin-top: 2rem;
        border-top: 2px solid #667eea;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .sidebar .stButton > button {
        width: 100%;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Configure Google AI (silently)
try:
    api_key = (
        os.getenv("GOOGLE_API_KEY") or 
        st.secrets.get("GOOGLE_API_KEY", None) or
        None
    )
    
    if not api_key:
        st.error("🔑 Please configure your Google API key")
        st.stop()
    
    client = genai.Client(api_key=api_key)
    
except Exception as e:
    st.error(f"❌ Configuration error: {str(e)}")
    st.stop()

def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDF files"""
    text = ""
    file_info = []
    
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            file_text = ""
            for page in pdf_reader.pages:
                file_text += page.extract_text() + "\n"
            text += f"\n--- Document: {pdf.name} ---\n{file_text}\n"
            file_info.append({"name": pdf.name, "pages": len(pdf_reader.pages)})
        except Exception as e:
            st.error(f"Error reading {pdf.name}: {str(e)}")
    
    return text, file_info

def get_text_chunks(text, chunk_size=1500, overlap=300):
    """Split text into overlapping chunks"""
    if not text.strip():
        return []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at sentence or paragraph end
        if end < len(text):
            break_points = [chunk.rfind('\n\n'), chunk.rfind('. '), chunk.rfind('\n')]
            break_point = max([bp for bp in break_points if bp > start + chunk_size // 2], default=-1)
            
            if break_point > 0:
                chunk = text[start:break_point + 1]
                start = break_point + 1 - overlap
            else:
                start = end - overlap
        else:
            start = end
            
        if chunk.strip():
            chunks.append(chunk.strip())
    
    return chunks

def get_embeddings(texts):
    """Get embeddings for text chunks"""
    embeddings = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, text in enumerate(texts):
        try:
            status_text.text(f"Processing chunk {i+1}/{len(texts)}...")
            result = client.models.embed_content(
                model="gemini-embedding-001",
                contents=text
            )
            embeddings.append(result.embeddings[0].values)
            progress_bar.progress((i + 1) / len(texts))
        except Exception as e:
            continue
    
    status_text.empty()
    progress_bar.empty()
    return embeddings

def save_embeddings(chunks, embeddings, file_info):
    """Save processed data"""
    try:
        data = {
            'chunks': chunks,
            'embeddings': embeddings,
            'file_info': file_info,
            'timestamp': datetime.now().isoformat()
        }
        with open('qubave_pdf_data.pkl', 'wb') as f:
            pickle.dump(data, f)
        return True
    except Exception as e:
        return False

def load_embeddings():
    """Load processed data"""
    try:
        with open('qubave_pdf_data.pkl', 'rb') as f:
            data = pickle.load(f)
        return data['chunks'], data['embeddings'], data.get('file_info', []), data.get('timestamp')
    except:
        return None, None, [], None

def find_relevant_context(question, chunks, embeddings, top_k=4):
    """Find relevant context from documents"""
    if not chunks or not embeddings:
        return ""
    
    try:
        # Get question embedding
        question_result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=question
        )
        question_embedding = np.array(question_result.embeddings[0].values).reshape(1, -1)
        
        # Calculate similarities
        chunk_embeddings = np.array(embeddings)
        similarities = cosine_similarity(question_embedding, chunk_embeddings)[0]
        
        # Get top-k most similar chunks
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        relevant_chunks = [chunks[i] for i in top_indices if similarities[i] > 0.3]  # Only include if reasonably relevant
        
        return "\n\n".join(relevant_chunks) if relevant_chunks else ""
        
    except Exception as e:
        return ""

def generate_response(question, context="", file_info=[]):
    """Generate natural AI response"""
    
    # Create a natural prompt that allows both document context and general knowledge
    if context:
        file_list = ", ".join([f["name"] for f in file_info]) if file_info else "uploaded documents"
        prompt = f"""You are Qubave AI, an intelligent assistant helping users analyze documents and answer questions. 

I have uploaded some documents ({file_list}) and here's the relevant content from them:

{context}

Question: {question}

Please provide a helpful, natural response. You can use both the information from the uploaded documents AND your general knowledge to give the most complete and useful answer. If you reference information from the documents, mention that it's from the uploaded files. Be conversational and helpful like a knowledgeable assistant.
"""
    else:
        prompt = f"""You are Qubave AI, an intelligent assistant. Please answer this question naturally and helpfully using your knowledge:

{question}

Note: No specific documents have been uploaded yet, so please answer based on your general knowledge."""
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return "I apologize, but I'm having trouble generating a response right now. Please try again."

def main():
    # Header with Qubave AI branding
    st.markdown("""
    <div class="main-header">
        <div class="qubave-logo">🤖 Qubave AI</div>
        <h2 style="margin: 0;">Intelligent Case Analysis System</h2>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Upload documents and ask anything - I'll help you understand and analyze your content</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load existing data
    chunks, embeddings, file_info, timestamp = load_embeddings()
    
    # Main chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Chat input
        user_question = st.text_area(
            "💬 Ask me anything about your documents:",
            height=100,
            placeholder="e.g., Summarize the key points from my documents, or ask any general question..."
        )
        
        if st.button("🚀 Ask ", type="primary"):
            if user_question:
                # Show user message
                st.markdown(f'<div class="user-message">👤 {user_question}</div>', unsafe_allow_html=True)
                
                with st.spinner("🤖 Qubave AI is thinking..."):
                    # Find relevant context if documents are available
                    context = find_relevant_context(user_question, chunks, embeddings) if chunks else ""
                    
                    # Generate response
                    response = generate_response(user_question, context, file_info)
                    
                    # Show AI response
                    st.markdown(f'<div class="ai-message">🤖 <strong>Qubave AI:</strong><br><br>{response}</div>', unsafe_allow_html=True)
    
    with col2:
        # Status panel
        if file_info:
            st.markdown("### 📁 Uploaded Documents")
            for info in file_info:
                st.info(f"📄 {info['name']}\n📄 {info.get('pages', 0)} pages")
            if timestamp:
                st.caption(f"Processed: {timestamp[:16]}")
        else:
            st.markdown("### 💡 How to use")
            st.info("1. Upload PDF documents\n2. Click 'Process Documents'\n3. Ask any questions!")
    
    # Sidebar for document management
    with st.sidebar:
        st.markdown("### 📂 Document Manager")
        
        pdf_docs = st.file_uploader(
            "Upload PDF Documents", 
            accept_multiple_files=True,
            type=['pdf'],
            help="Upload one or more PDF files to analyze"
        )
        
        if pdf_docs:
            st.success(f"✅ {len(pdf_docs)} file(s) ready")
            for pdf in pdf_docs:
                st.caption(f"📄 {pdf.name}")
        
        if st.button("🔄 Process Documents", use_container_width=True):
            if not pdf_docs:
                st.error("Please upload PDF files first")
            else:
                with st.spinner("🔄 Processing your documents..."):
                    raw_text, file_info = get_pdf_text(pdf_docs)
                    
                    if raw_text:
                        text_chunks = get_text_chunks(raw_text)
                        
                        if text_chunks:
                            embeddings = get_embeddings(text_chunks)
                            
                            if embeddings and save_embeddings(text_chunks, embeddings, file_info):
                                st.success("✅ Documents processed successfully!")
                                st.balloons()
                                st.rerun()
                            else:
                                st.error("Failed to process documents")
                        else:
                            st.error("No content found in documents")
                    else:
                        st.error("Could not extract text from documents")
        
        st.markdown("---")
        
        # Clear data option
        if st.button("🗑️ Clear All Data", use_container_width=True):
            try:
                if os.path.exists('qubave_pdf_data.pkl'):
                    os.remove('qubave_pdf_data.pkl')
                st.success("✅ All data cleared!")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")
        
        st.markdown("---")
        
        # About section
        st.markdown("### ℹ️ About")
        st.info("This tool uses advanced AI to help you understand and analyze your documents.")
    
    # Footer
    st.markdown("""
    <div class="footer-brand">
        <p style="margin: 0; color: #667eea; font-weight: bold;">Powered by Qubave AI</p>
        <p style="margin: 0; font-size: 0.8rem; color: #666;">Intelligent Document Analysis • Advanced AI Technology</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()