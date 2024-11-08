## **RAG Langchain Model with OpenAI API** 

### _Project Overview_
This repository contains the code and documentation for a Retrieval Augmented Generation (RAG) model, developed by Dani Siaj and Carlos Rodríguez. This model enables users to upload a PDF document, ask questions, and receive coherent, complete, and relevant responses generated by an integrated large language model (LLM).

The RAG model dynamically generates a prompt from the user's query, incorporating instructions, context, and restrictions to create specific, contextually aware responses.

### _Content_
The uploaded PDF is a 9-page document containing information on food allergies, symptoms, and management, sourced from the American College of Allergy, Asthma, and Immunology (ACAAI). This document includes only textual content—no tables or images are present.

### _Model Architecture_
#### Model Selection
The model architecture is centered around OpenAIEmbeddings API as the text transformer. Key libraries used include:

* Langchain: For text extraction and model chaining.
* Chroma DB: To create and manage the vector store.

#### Components
* Document Loader: PyPDFLoader handles document uploads and text parsing.
* Embeddings: OpenAIEmbeddings transforms text into vector representations.
* Text Extraction: RecursiveTextCharacterSplitter and ChromaDB handle text processing and vectorization.

### _Chain Architecture_
### * _Retrieval of Information_: 
User queries retrieve a set of k documents (where k=3 in the code) from the ChromaDB vector store using similarity_search().
### * _Prompt Engineering_:
* A context is built using the selected documents.
* This context is passed to the dynamic prompt-generating function.
* A specific, context-aware prompt is created based on the user’s query.
### * _LLM Implementation_: 
The prompt is sent to the LLM via the OpenAI API to generate the desired response.
### _Model Evaluation_
A second LLM model is used as a "judge" to evaluate the generated responses based on the following criteria:

 * Relevance(0-5)
 * Accuracy(0-5)
 * Completeness(0-5)
 * Clarity(0-5)

Through prompt engineering, a dedicated evaluation prompt is used to assess the quality of each response.

### _Streamlit App_
The model is deployed on Streamlit, providing a user-friendly interface. Users can input questions and receive responses formatted in Markdown, followed by the LLM-based evaluation. This design enhances user experience by providing both a direct answer and an automated quality assessment.

### _Conclusions_
Conclusion 1: The RAG model demonstrated high efficiency in terms of response time and relevance to user queries.
Conclusion 2: The limited size of the document restricts extensive testing. Future evaluations will include larger files for a more comprehensive assessment.

### _Repository_
* Data folder: where the PDF documents and the Chroma DB is stored
* main.py where all the code is organized
* Pptx presentation
* ReadMe.md
* Requirements.txt with all the neccessary libraries for this project
* Streamlit_RAG.py to deploy the code in Streamlit platform and test it in an application