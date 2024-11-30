import os
import json
import concurrent.futures
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

UPLOAD_DIR = "uploads"
JSON_DIR = "json_data"
CHAT_HISTORY_FILE = "chat_history.json"

def save_uploaded_file(uploaded_file):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def list_files():
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)
    return [f for f in os.listdir(UPLOAD_DIR) if f.endswith(".pdf")]

def delete_file(file_name):
    file_path = os.path.join(UPLOAD_DIR, file_name)
    if os.path.exists(file_path):
        os.remove(file_path)
        json_file_path = os.path.join(JSON_DIR, f"{os.path.splitext(file_name)[0]}.json")
        if os.path.exists(json_file_path):
            os.remove(json_file_path)
        return True
    return False

@st.cache_resource
def process_all_pdfs():
    documents = []
    files = list_files()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_single_pdf, os.path.join(UPLOAD_DIR, file_name), file_name) for file_name in files]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                documents.extend(result)
    return documents

def process_single_pdf(file_path, file_name):
    try:
        pdf_reader = PdfReader(file_path)
        data = [{"text": page.extract_text() or "", "page_num": page_num + 1, "file_name": file_name}
                for page_num, page in enumerate(pdf_reader.pages)]

        # Save extracted data to a JSON file
        os.makedirs(JSON_DIR, exist_ok=True)
        json_file_path = os.path.join(JSON_DIR, f"{os.path.splitext(file_name)[0]}.json")
        with open(json_file_path, "w") as json_file:
            json.dump(data, json_file, indent=4)

        st.write(f"Extracted and saved JSON for {file_name} at {json_file_path}")  # Debugging line

        return data
    except Exception as e:
        st.error(f"Error processing {file_name}: {e}")
        return []

@st.cache_resource
def create_knowledge_base():
    documents = []
    if not os.path.exists(JSON_DIR):
        os.makedirs(JSON_DIR)
    
    # Load all JSON files and append to documents
    for json_file_name in os.listdir(JSON_DIR):
        if json_file_name.endswith(".json"):
            json_file_path = os.path.join(JSON_DIR, json_file_name)
            with open(json_file_path, "r") as json_file:
                documents.extend(json.load(json_file))

    if documents:
        texts = [
            f"[Page {doc['page_num']} in {doc['file_name']}]: {doc['text']}" for doc in documents if doc['text'].strip() != ""
        ]
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_text("\n".join(texts))

        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        st.write("Knowledge base created successfully with the extracted JSON data.")  # Debugging line
        return knowledge_base
    return None

def load_chat_history():
    if not os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "w") as f:
            json.dump([], f)

    try:
        with open(CHAT_HISTORY_FILE, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        with open(CHAT_HISTORY_FILE, "w") as f:
            json.dump([], f)
        return []

def save_chat_history(chat_history):
    with open(CHAT_HISTORY_FILE, "w") as f:
        json.dump(chat_history, f, indent=4)

def greet_user(user_input):
    greetings = ["hi", "hello", "hey"]
    if any(greeting in user_input.lower() for greeting in greetings):
        return "Hi, this is Jack. How can I assist you today?"
    return None

def is_follow_up_question(new_question, last_question):
    follow_up_keywords = ["what", "more about", "can you elaborate", "tell me more"]
    return any(keyword in new_question.lower() for keyword in follow_up_keywords)

def redirect_to_topic(user_input):
    unrelated_topics = ["weather", "sports", "movies", "music", "news"]
    if any(topic in user_input.lower() for topic in unrelated_topics):
        return "That's an interesting topic! However, let's get back to how I can help you with your PDFs or the extracted knowledge base."
    return None

def generate_smart_response(user_question, docs, llm):
    if docs:
        prompt_template = """You are a helpful assistant that provides clear and concise answers. Use the following information to answer the user's question as accurately as possible. Be informative but keep it conversational.
Context: {context}
Question: {question}
Answer:"""
        prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

        chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)

        response = chain.run(input_documents=docs, question=user_question)

        if response.strip():
            # Breakdown response into bullet points if appropriate
            response_lines = response.split("\n")
            if len(response_lines) > 1:
                response = "\n- " + "\n- ".join([line.strip() for line in response_lines if line.strip()])
            else:
                response = "- " + response.strip()
            return response.strip()
        else:
            return "I couldn't find any relevant information in the uploaded PDFs. Could you provide more details or ask something else?"
    else:
        return "I couldn't find any relevant documents to answer your question. Please make sure your question is related to the content of the uploaded PDFs."

def main():
    load_dotenv()

    st.set_page_config(
        page_title="The Anchor Builders",
        page_icon="🌟",
        layout="wide",
    )

    st.sidebar.title("Navigation")
    st.sidebar.subheader("Manage your PDFs")
    files = list_files()

    if files:
        with st.sidebar.expander("Uploaded Files", expanded=True):
            for file in files:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(file)
                with col2:
                    if st.button("🗑️", key=f"delete-{file}"):
                        if delete_file(file):
                            st.sidebar.success(f"Deleted {file}")
                            st.experimental_rerun()
                        else:
                            st.sidebar.error(f"Failed to delete {file}")
    else:
        st.sidebar.info("No files uploaded yet.")

    pdf = st.sidebar.file_uploader("Upload a PDF", type="pdf", accept_multiple_files=True)
    if pdf:
        for uploaded_file in pdf:
            save_uploaded_file(uploaded_file)
            st.sidebar.success(f"Uploaded: {uploaded_file.name}")
        process_all_pdfs()  # Explicitly call to process PDFs immediately after upload
        st.experimental_rerun()

    # Load chat history
    chat_history = load_chat_history()

    st.sidebar.subheader("Chat History")
    if chat_history:
        for index, chat in enumerate(chat_history):
            with st.sidebar.expander(f"Q: {chat['question']}"):
                st.markdown(f"**A:** {chat['answer']}")
                if st.button("🗑️ Delete", key=f"delete-chat-{index}"):
                    chat_history.pop(index)
                    save_chat_history(chat_history)
                    st.experimental_rerun()
        if st.sidebar.button("Delete All Chats"):
            with open(CHAT_HISTORY_FILE, "w") as f:
                json.dump([], f)
            st.experimental_rerun()
    else:
        st.sidebar.info("No chat history yet.")

    st.title("📜 PDF Knowledge Assistant")
    st.markdown("Upload your PDFs, ask questions, and get precise answers with citations.")

    knowledge_base = create_knowledge_base()

    if knowledge_base:
        st.subheader("💬 Ask Your Questions")

        user_question = st.text_input(
            "Type your question here:",
            placeholder="e.g., What are the height restrictions for ADUs?",
            key="user_input",
        )

        if user_question:
            greeting_response = greet_user(user_question)
            if greeting_response:
                st.info(greeting_response)
                return

            redirection = redirect_to_topic(user_question)
            if redirection:
                st.info(redirection)
                return

            if chat_history:
                last_chat = chat_history[-1]
                last_question = last_chat["question"]
                if is_follow_up_question(user_question, last_question):
                    user_question = f"{last_question} {user_question}"

            docs = knowledge_base.similarity_search(user_question, k=3)
            llm = OpenAI()

            response = generate_smart_response(user_question, docs, llm)

            chat_history.append({"question": user_question, "answer": response})
            save_chat_history(chat_history)

            st.subheader("Current Conversation")
            st.markdown(f"**Q:** {user_question}")
            st.markdown(response)
            st.markdown("---")

            if docs:
                st.subheader("Source Information")
                for doc in docs:
                    file_name = doc.metadata.get('file_name', 'Unknown File')
                    page_num = doc.metadata.get('page_num', 'Unknown Page')
                    st.markdown(f"- **{file_name}**, Page {page_num}")

                st.info("Extracted from your uploaded PDFs. Cross-reference for accuracy when needed.")
    else:
        st.warning("No data available. Please upload PDF files to begin.")

if __name__ == "__main__":
    main()
