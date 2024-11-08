import streamlit as st
import pandas as pd
import numpy as np
import os
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from IPython.display import display, Markdown
from uuid import uuid4
from langchain_core.documents import Document


openai_api_key = st.sidebar.text_input('OpenAI API Key')
#### Code ####


# API KEY from OpenAI
API_KEY = openai_api_key

# Initialize client from OpenAI
client = OpenAI(api_key = API_KEY)

# Embeddings object from OpenAI
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key = API_KEY)

##### 2. Functions to process data #####

# PDF loader function
def load_and_split_pdf(file_path):
    """
    This function uses the langchain library to load a PDF document and split it into pages.

    Arguments: path for pdf document
    """
    pages = PyPDFLoader(file_path).load_and_split() # Split the document in pages
    print(len(pages))
    return pages

# Text splitter function
def text_splitter_pages(pages):
    """
    This function takes uses the RecursiveCharacterTextSplitter from langchain 
    to split the text into chunks of 700 characters with an overlap of 50

    Arguments: str
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
    chunks = text_splitter.split_documents(pages) # Split the pages into chunks

    return chunks

# Create embeddings and store the chroma database 
def get_or_create_vectorstore(embeddings):
    """
    This function initializes the vector store from Chroma, or creates a new one if it does not exist

    Arguments: chunks: str; embeddings model
    """
    db = Chroma(
    collection_name="nursing_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_nursing_db",  # Where to save data locally, remove if not necessary
    )

    return db

# Load the chroma database if it already exists
def load_documents_to_vectorstore(chunks, db, batch_size=100):
    """
    This function loads the existing vector store from the path

    """
    
    print('embedding')
    total_chunks = len(chunks)
    for i in range(0, total_chunks, batch_size):
        batch = chunks[i:i + batch_size]
        uuids = [str(uuid4()) for _ in range(len(batch))]
        db.add_documents(documents=batch, ids=uuids)
        print(f'Uploaded batch {i // batch_size + 1}/{(total_chunks // batch_size) + 1}')
    print('All document uploads completed')

    return db

##### 3. Functions to generate the prompt and retrieve the answer from our LLM #####

# User query function
def get_query():
    """
    This function prompts the user to type a question
    """
    query = input("Ask a question about allergies: ")

    return query

# Similarity Search function
def similarity_search(db, query):
    """
    This function takes the query from the user and matches it with the 3 more similar embedding from the Chroma db

    Arguments: db: vector store, user_question: str
    """
    docs = db.similarity_search(query, k=3)
    return docs

# Build context for prompt function
def _get_document_context(docs):
    """
    This function builds a context paragraph for the prompt, using the results from the similarity search function

    Arguments: str
    """
    context = '\n'
    for doc in docs:
        context += '\nContext:\n'
        context += doc.page_content + '\n\n'

    return context

# Dynamic prompt function
def generate_prompt_from_user_query(query, context):
    """
    This functions uses a template to generate a dynamic prompt that can be adapted to the user's query

    Arguments: user_question: str, docs :str
    """
    prompt = f"""
        INTRODUCTION
        You are a knowledgeable assistant trained to answer questions about allergies, symptoms, and management strategies. Your responses should be clear, concise, and focused on accurate information. Always in markdown format.

        The user asked: "{query}"

        CONTEXT
        Technical documentation for allergies, symptoms, and management of allergen ingestion:
        '''
        {_get_document_context(context)}
        '''

        RESTRICTIONS
        Always refer to products or allergens by their specific names as mentioned in the documentation.
        Stick to facts and provide clear, evidence-based responses; avoid opinions or interpretations.
        Only respond if the answer can be found within the context. If not, let the user know that the information is not available.
        Do not engage in topics outside allergies, symptoms, and related health matters. Avoid humor, sensitive topics, and speculative discussions.
        If the user’s question lacks sufficient details, request clarification rather than guessing the answer. For example, if the user does not ask anything related to allergies, allergies symptoms, or allergies management, you should request clarification.
        EXAMPLE:
            example 1:
                User: 'I ate eggs'
                Agent: 'I hope they tasted amazing. Are you allergic to eggs?'

            example 2: 
                User: 'I think I have an allergy to eggs'
                Agent: 'Egg allergies are common and can cause a range of symptoms, from mild to more severe reactions. Here are some typical signs and management steps:
                        Symptoms of an Egg Allergy
                        Mild Reactions: Skin reactions like hives, eczema, or redness; digestive issues such as cramps, nausea, or vomiting; and runny nose or sneezing.
                        Severe Reactions (Anaphylaxis): Difficulty breathing, swelling of the throat, rapid pulse, dizziness, or loss of consciousness.
                        If you experience severe symptoms, you should seek medical help immediately, as anaphylaxis requires prompt treatment.

                        Management and Avoidance Tips
                        Avoid Egg-Based Foods: Eggs can be hidden in foods, so check labels for ingredients like “albumin” or “lysozyme” that indicate eggs.
                        Consider Egg Substitutes: For baking, substitutes like applesauce, banana, or commercial egg replacers can be helpful.
                        Discuss with Your Doctor: They may suggest an allergy test to confirm the allergy or advise on an emergency plan, such as carrying an epinephrine auto-injector if needed.
                        If you’re experiencing ongoing symptoms or suspect an allergy, consulting with an allergist is recommended for personalized advice and treatment.
        TASK
        Provide a direct answer based on the user’s question, if possible.
        Guide the user to relevant sections of the documentation if additional context is needed.
        always provide answers in Markdown format
        
        CONVERSATION:
        User: {query}
        Agent:
        """
    return prompt

# Get response from LLM function
def get_response_from_client(prompt):
    """
    This function initiliazes an OpenAI chat to generate the response to a query.
    Arguments: prompt: str
    """
    client = OpenAI(api_key = API_KEY)

    messages = [{'role':'user', 'content':prompt}]
    model_params = {'model': 'gpt-4o-mini', 'temperature': 0.4, 'max_tokens': 200}
    completion = client.chat.completions.create(messages=messages, **model_params, timeout=120)

    answer = completion.choices[0].message.content

    return answer

# COMBINE THE PREVIOUS FUNCTIONS
def get_response(db, query):
    context = similarity_search(db, query)
    prompt = generate_prompt_from_user_query(query, context)
    answer = get_response_from_client(prompt)

    return answer




##### 4. Functions for OpenAI API #####

# Initialize OpenAI assistant
def build_LLM_as_a_judge(client):
    """
    This function initializes the assistant from Open AI that will act as the judge for our model's answers

    Arguments: client: OpenAI class
    """

    judge = client.beta.assistants.create(
        name="LLM as a Judge",
        instructions="You are an expert evaluator for answers generated by other LLMs",
        model="gpt-4o-mini",
        tools=[{"type": "file_search"}]
        )

    return judge

# Create and upload vector store for OpenAI assistant
def vector_store_for_assistant(client, file_path):
    """
    this function creates a vector store and uploades it to OpenAI for our assistant to use

    Argument: client: OpenAI class, file_path: str
    """

    # Create Vector Store
    vector_store = client.beta.vector_stores.create(name="allergies_document")

    # Create paths and streams from the documents
    file_paths = [file_path]
    file_streams = [open(path, "rb") for path in file_paths]

    # Store the bath of files and upload it to OpenAI
    file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
    vector_store_id=vector_store.id, files=file_streams
    )

    # Update the assistant so it can use the new files
    assistant = client.beta.assistants.update(
        assistant_id=assistant.id,
        tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}},
        )
    
    return False # to avoid repeating the process

# Generates a new prompt based on the question from the user and the answer generated by our model, for evaluation   
def generate_prompt_for_eval(query, answer):
    """
    This function creates a dynamic prompt that will be used to ask our LLM (in this case, OpenAI)
    to evaluate our model's answer.

    Arguments: query: str, answer: str
    """
    prompt_for_eval = f"""
        Task:
        You are an expert evaluator tasked with assessing the quality of responses generated by an AI model. 
        The model takes a question and provides an answer with a maximum length of 200 tokens. 
        Please evaluate the answer according to the Evaluation Criteria provided below, and provide 4 different scores, one score for each different criteria from 0 to 5,
        with 0 being completely incorrect or irrelevant and 5 being exceptionally accurate, coherent, and comprehensive.

        Evaluation Criteria:
        Relevance: Is the answer directly relevant to the question? Provide a score from 0 to 5
        Accuracy: Does the answer provide correct and factual information? Provide a score from 0 to 5
        Completeness: Does the answer sufficiently cover the main points without missing key information? Provide a score from 0 to 5
        Clarity: Is the answer clear, easy to understand, and well-structured? Provide a score from 0 to 5

        Scoring Scale:
        5: Excellent – Highly accurate, relevant, and complete answer with clear, coherent language.
        4: Good – Mostly accurate and relevant answer, with minor omissions or slight clarity issues.
        3: Adequate – Provides some relevant information but may lack accuracy, completeness, or clarity in parts.
        2: Poor – Limited relevance or accuracy, missing key points, or difficult to understand.
        1: Very Poor – Largely irrelevant or incorrect answer.
        0: No relevance – Completely off-topic or nonsensical answer.

        Format: Please provide the following:
        Relevance Score: (0-5)
        Accuracy Score: (0-5)
        Completeness Score: (0-5)
        Clarity Score: (0-5)
        Brief Justification: Describe why you assigned these scores based on relevance, accuracy, completeness, and clarity.
        Here is the Question: {query}
        And here is the Answer: {answer}

        Thank you."""
    
    return prompt_for_eval

# Our LLM evaluates our model's answer and generates a score with an explanation
def get_evaluation_from_LLM_as_a_judge(client, prompt_for_eval):
    """
    This function calls the LLM designated to be the judge. The judge will evaluate the answer provided by our model,
    and it will return 4 different scores, evaluating the answer using the following criteria:

    Evaluation Criteria:
        Relevance: Is the answer directly relevant to the question? 
        Accuracy: Does the answer provide correct and factual information? 
        Completeness: Does the answer sufficiently cover the main points without missing key information? 
        Clarity: Is the answer clear, easy to understand, and well-structured? 

    Arguments: client: OpenAI object; prompt_for_eval: str
    """

    messages = [{'role':'user', 'content':prompt_for_eval}] 
    model_params = {'model': 'gpt-4o-mini', 'temperature': 0.4, 'max_tokens': 200}
    completion = client.chat.completions.create(messages=messages, **model_params, timeout=120)

    evaluation = completion.choices[0].message.content

    return evaluation


page_bg_img = '''
<style>
body {
background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
background-size: cover;
}
</style>
'''



st.markdown(page_bg_img, unsafe_allow_html=True)

st.title("Food Allergies App")
st.write('This app will tell you all you need to know about food allergies!')

st.header("Ask a question about food allergies")
#user_query = st.text_input("type your question here", "e.g.: List all the food allergies!")
with st.form('my_form'):
    uploaded_file = st.file_uploader("Upload a document", type=["pdf", "docx"])

    user_query = st.text_area('type your question here', 'e.g.: List all the food allergies!')
    submitted = st.form_submit_button('Submit')
    if submitted:
        st.write('Generating response...')
        ##### Check if the vector store exists #####
        db = get_or_create_vectorstore(embeddings)

        query = user_query

        ##### Get response from LLM #####
        answer = get_response(db, query) # this line generates the dynamic promtp for RAG, and calls the LLm for an answer

        ##### Evaluate response #####
        st.write('Evaluating response...')
        prompt_for_eval = generate_prompt_for_eval(query, answer) 
        evaluation = get_evaluation_from_LLM_as_a_judge(client, prompt_for_eval)

        ## 5. Display answer
        dash_line = '------------------------'
        st.write(dash_line)

        st.write(f"My Question: {query}")
        st.write(dash_line)
        st.markdown(answer)
        st.write(dash_line)
        st.write("Evaluation from LLM")
        st.markdown(evaluation)








