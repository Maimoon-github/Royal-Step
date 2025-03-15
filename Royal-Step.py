# import os
# import streamlit as st

# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.chains import RetrievalQA

# from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import PromptTemplate
# from langchain_huggingface import HuggingFaceEndpoint

# ## Uncomment the following files if you're not using pipenv as your virtual environment manager
# #from dotenv import load_dotenv, find_dotenv
# #load_dotenv(find_dotenv())


# DB_FAISS_PATH="vectorstore/db_faiss"
# @st.cache_resource
# def get_vectorstore():
#     embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
#     db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
#     return db


# def set_custom_prompt(custom_prompt_template):
#     prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
#     return prompt


# def load_llm(huggingface_repo_id, HF_TOKEN):
#     llm=HuggingFaceEndpoint(
#         repo_id=huggingface_repo_id,
#         temperature=0.5,
#         model_kwargs={"token":HF_TOKEN,
#                       "max_length":"512"}
#     )
#     return llm


# def main():
#     st.title("Ask Chatbot!")

#     if 'messages' not in st.session_state:
#         st.session_state.messages = []

#     for message in st.session_state.messages:
#         st.chat_message(message['role']).markdown(message['content'])

#     prompt=st.chat_input("Pass your prompt here")

#     if prompt:
#         st.chat_message('user').markdown(prompt)
#         st.session_state.messages.append({'role':'user', 'content': prompt})

#         CUSTOM_PROMPT_TEMPLATE = """
#                 Use the pieces of information provided in the context to answer user's question.
#                 If you dont know the answer, just say that you dont know, dont try to make up an answer. 
#                 Dont provide anything out of the given context

#                 Context: {context}
#                 Question: {question}

#                 Start the answer directly. No small talk please.
#                 """
        
#         HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"
#         HF_TOKEN=os.environ.get("HF_TOKEN")

#         try: 
#             vectorstore=get_vectorstore()
#             if vectorstore is None:
#                 st.error("Failed to load the vector store")

#             qa_chain=RetrievalQA.from_chain_type(
#                 llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
#                 chain_type="stuff",
#                 retriever=vectorstore.as_retriever(search_kwargs={'k':3}),
#                 return_source_documents=True,
#                 chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
#             )

#             response=qa_chain.invoke({'query':prompt})

#             result=response["result"]
#             source_documents=response["source_documents"]
#             result_to_show=result+"\nSource Docs:\n"+str(source_documents)
#             #response="Hi, I am MediBot!"
#             st.chat_message('assistant').markdown(result_to_show)
#             st.session_state.messages.append({'role':'assistant', 'content': result_to_show})

#         except Exception as e:
#             st.error(f"Error: {str(e)}")

# if __name__ == "__main__":
#     main()





# import asyncio
# import sys
# if sys.platform == "win32" and sys.version_info >= (3, 8):
#     asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# import os
# import streamlit as st
# from langchain_community.embeddings import HuggingFaceEmbeddings
# import os
# # from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.chains import RetrievalQA
# from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import PromptTemplate
# from langchain_huggingface import HuggingFaceEndpoint
# from dotenv import load_dotenv, find_dotenv

# # Load environment variables
# load_dotenv(find_dotenv())

# DB_FAISS_PATH = "vectorstore/db_faiss"

# @st.cache_resource
# def get_vectorstore():
#     embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
#     db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
#     return db

# def set_custom_prompt():
#     return PromptTemplate(
#         template="""**Role:**  
#     You are an AI-powered customer service assistant designed to provide accurate, helpful, and polite responses to users’ inquiries about products, services, policies, and troubleshooting issues.

#     ---

#     **Guidelines:**
#     1. **Politeness & Professionalism:** Greet users, maintain a respectful tone, and end conversations politely.
#     2. **Concise Responses:** Provide clear, direct answers.
#     3. **Personalization:** Address users by name if provided.
#     4. **Handling Unclear Queries:** Ask for clarification when needed.
#     5. **Escalation:** If needed, guide users to human support.

#     **Context:** {context}
#     **Question:** {question}

#     Only respond with information from the given context. If uncertain, say you don't know.""",
#         input_variables=["context", "question"]
#     )

# def load_llm():
#     return HuggingFaceEndpoint(
#         repo_id="mistralai/Mistral-7B-Instruct-v0.3",
#         temperature=0.5,
#         model_kwargs={"max_length": auto}
#     )

# def format_source_documents(docs):
#     formatted = []
#     for doc in docs:
#         source = os.path.basename(doc.metadata['source'])
#         content = doc.page_content.replace('\n', ' ').strip()
#         formatted.append(f"📄 {source} (Page {doc.metadata.get('page', 'N/A')}): {content[:250]}...")
#     return "\n\n".join(formatted)

# def main():
#     st.title("MediBot - Medical Knowledge Assistant")
    
#     if 'messages' not in st.session_state:
#         st.session_state.messages = [{"role": "assistant", "content": "How can I help you with medical information today?"}]

#     for message in st.session_state.messages:
#         st.chat_message(message['role']).markdown(message['content'])

#     if prompt := st.chat_input("Enter your medical question..."):
#         st.chat_message("user").markdown(prompt)
#         st.session_state.messages.append({"role": "user", "content": prompt})

#         try:
#             # Initialize components
#             vectorstore = get_vectorstore()
#             qa_chain = RetrievalQA.from_chain_type(
#                 llm=load_llm(),
#                 chain_type="stuff",
#                 retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
#                 return_source_documents=True,
#                 chain_type_kwargs={'prompt': set_custom_prompt()}
#             )

#             # Process query
#             response = qa_chain.invoke({'query': prompt})
#             answer = response["result"]
#             sources = format_source_documents(response["source_documents"])

#             # Display structured response
#             st.chat_message("assistant").markdown(answer)
            
#             # Show sources in expandable section
#             with st.expander("View Supporting Documents"):
#                 st.markdown(sources)

#             st.session_state.messages.append({"role": "assistant", "content": answer})

#         except Exception as e:
#             st.error(f"Error processing request: {str(e)}")

# if __name__ == "__main__":
#     main()







import asyncio
import sys
import os
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv, find_dotenv

# Ensure compatibility with Windows event loop
if sys.platform == "win32" and sys.version_info >= (3, 8):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Load environment variables
load_dotenv(find_dotenv())

# Constants
DB_FAISS_PATH = "vectorstore/db_faiss"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

def set_custom_prompt():
    return PromptTemplate(
        template="""
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
        """,
        input_variables=["context", "question"]
    )

def load_llm():
    return HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO_ID,
        temperature=0.5,
        model_kwargs={"max_length": 512}
    )

def format_source_documents(docs):
    formatted = []
    for doc in docs:
        source = os.path.basename(doc.metadata.get('source', 'Unknown Source'))
        content = doc.page_content.replace('\n', ' ').strip()
        formatted.append(f"📄 {source} (Page {doc.metadata.get('page', 'N/A')}): {content[:250]}...")
    return "\n\n".join(formatted)

def main():
    st.title("INFO Provider - Royal-Step Knowledge Assistant")
    
    if 'messages' not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How can I help you with any specific information today?"}]

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    if prompt := st.chat_input("Enter your medical question..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        try:
            vectorstore = get_vectorstore()
            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt()}
            )

            response = qa_chain.invoke({'query': prompt})
            answer = response.get("result", "I couldn't find an answer to your question.")
            sources = format_source_documents(response.get("source_documents", []))

            st.chat_message("assistant").markdown(answer)
            
            if sources:
                with st.expander("View Supporting Documents"):
                    st.markdown(sources)

            st.session_state.messages.append({"role": "assistant", "content": answer})

        except Exception as e:
            st.error(f"Error processing request: {str(e)}")

if __name__ == "__main__":
    main()
