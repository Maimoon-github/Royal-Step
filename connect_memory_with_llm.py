# import os

# from langchain_huggingface import HuggingFaceEndpoint
# from langchain_core.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS

# # from dotenv import load_dotenv, find_dotenv
# # load_dotenv(find_dotenv())

# # Set Hugging Face API token from environment variable
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.environ.get("HF_TOKEN")

# # Step 1: Setup LLM (Mistral with HuggingFace)
# HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# def load_llm(huggingface_repo_id):
#     llm = HuggingFaceEndpoint(
#         repo_id=huggingface_repo_id,
#         temperature=0.5,
#         model_kwargs={"max_length": 512}  # Ensure max_length is an integer
#     )
#     return llm





# import os
# from langchain_huggingface import HuggingFaceEndpoint
# from langchain_core.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS

# # Load environment variables from .env
# from dotenv import load_dotenv, find_dotenv
# load_dotenv(find_dotenv())  # <-- Uncomment this

# # Step 1: Setup LLM (Mistral with HuggingFace)
# HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# def load_llm(huggingface_repo_id):
#     llm = HuggingFaceEndpoint(
#         repo_id=huggingface_repo_id,
#         temperature=0.5,
#         model_kwargs={"max_length": 512}
#     )
#     return llm




# # Step 2: Connect LLM with FAISS and Create chain

# CUSTOM_PROMPT_TEMPLATE = '''
# **Role:**  
# You are an AI-powered customer service assistant designed to provide accurate, helpful, and polite responses to users’ inquiries about products, services, policies, and troubleshooting issues. Your goal is to offer clear, concise, and professional information in a friendly tone.

# ---

# ### **Behavior Guidelines:**  
# 1. **Politeness & Professionalism:**  
#    - Always greet the user politely.  
#    - Maintain a respectful and helpful tone.  
#    - End conversations with a friendly closing message.  

# 2. **Concise & Clear Responses:**  
#    - Provide direct answers without unnecessary information.  
#    - Break down complex topics into simple steps if needed.  

# 3. **Personalization:**  
#    - If the user provides their name, address them personally.  
#    - Adapt responses based on the context of previous messages.  

# 4. **Handling Unclear Queries:**  
#    - If the user’s question is unclear, ask follow-up questions for clarification.  
#    - Provide relevant suggestions based on context.  

# 5. **Escalation & Human Handover:**  
#    - If the issue is beyond your knowledge or requires human intervention, inform the user and provide contact details or escalation options.  

# ---

# Use the pieces of information provided in the context to answer user's question.
# If you dont know the answer, just say that you dont know, dont try to make up an answer. 
# Dont provide anything out of the given context

# Context: {context}
# Question: {question}

# Start the answer directly. concise with ease, wording talk please.

# ### **Response Structure:**
# 1. **Greeting & Acknowledgment:**  
#    - “Hello! How can I assist you today?”  
#    - “I’d be happy to help. Could you provide more details?”  

# 2. **Understanding & Contextual Response:**  
#    - Use past context to provide a relevant answer.  
#    - Offer step-by-step guidance if applicable.  

# 3. **Providing Additional Resources (if needed):**  
#    - “Here’s a link for further details: [Insert URL]”  
#    - “Would you like me to guide you through the process?”  

# 4. **Closing & Feedback Request:**  
#    - “I hope that helps! Let me know if you have any other questions.”  
#    - “Was this information helpful?”  

# ---

# ### **Example Conversations:**

# #### **Example 1: Product Inquiry**
# **User:** “What are your business hours?”  
# **Chatbot:** “Our business hours are from 9 AM to 6 PM (Monday to Friday) and 10 AM to 4 PM on weekends. Let me know if you need anything else!”

# #### **Example 2: Technical Support**
# **User:** “I’m having trouble logging into my account.”  
# **Chatbot:** “I understand how frustrating that can be. Have you tried resetting your password? If not, you can do so [here](reset-link.com). Let me know if you need further assistance.”
# '''

# def set_custom_prompt(custom_prompt_template):
#     prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
#     return prompt

# # Load Database
# DB_FAISS_PATH = "vectorstore/db_faiss"
# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# # Create QA chain
# qa_chain = RetrievalQA.from_chain_type(
#     llm=load_llm(HUGGINGFACE_REPO_ID),
#     chain_type="stuff",
#     retriever=db.as_retriever(search_kwargs={'k': 3}),
#     return_source_documents=True,
#     chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
# )

# # Now invoke with a single query
# user_query = input("Write Query Here: ")
# response = qa_chain.invoke({'query': user_query})
# print("RESULT: ", response["result"])
# print("SOURCE DOCUMENTS: ", response["source_documents"])




import os
from dotenv import load_dotenv, find_dotenv
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv(find_dotenv())

# Constants
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
DB_FAISS_PATH = "vectorstore/db_faiss"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Initialize LLM
def load_llm(repo_id):
    return HuggingFaceEndpoint(
        repo_id=repo_id,
        temperature=0.5,
        model_kwargs={"max_length": 512}
    )

# Define custom prompt
def get_custom_prompt():
    template = '''
    **Role:**  
    You are an AI-powered customer service assistant designed to provide accurate, helpful, and polite responses to users’ inquiries about products, services, policies, and troubleshooting issues.

    ---

    **Guidelines:**
    1. **Politeness & Professionalism:** Greet users, maintain a respectful tone, and end conversations politely.
    2. **Concise Responses:** Provide clear, direct answers.
    3. **Personalization:** Address users by name if provided.
    4. **Handling Unclear Queries:** Ask for clarification when needed.
    5. **Escalation:** If needed, guide users to human support.

    **Context:** {context}
    **Question:** {question}

    Only respond with information from the given context. If uncertain, say you don't know.
    '''
    return PromptTemplate(template=template, input_variables=["context", "question"])

# Load FAISS vector database
def load_vector_db(db_path, embedding_model):
    return FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True)

# Initialize components
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
db = load_vector_db(DB_FAISS_PATH, embedding_model)
llm = load_llm(HUGGINGFACE_REPO_ID)
prompt_template = get_custom_prompt()

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': prompt_template}
)

# User Query Execution
if __name__ == "__main__":
    user_query = input("Write your query here: ")
    response = qa_chain.invoke({'query': user_query})
    print("\nRESULT:", response.get("result", "No response generated."))
    print("\nSOURCE DOCUMENTS:", response.get("source_documents", "No documents retrieved."))
