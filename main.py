import os
import re
import json
import pdfplumber
import requests
import chromadb
from dotenv import load_dotenv

load_dotenv()

def extract_txt_create_chunks():
    pdfs=os.listdir('pdfs') #get all the pdf names inside pdfs directory

    for pdf in pdfs: #iterate all pdfs 1 by 1 fro extraction and chunking

        chunks=[] #empty list which will hold dictionary of pdf wise chunks with meta_data

        with pdfplumber.open(f"pdfs/{pdf}") as file: #Using pdfplumber library to extract the text from pdf

            
            for page_number,page in enumerate(file.pages): #Iterating over each page in a pdf
                
                text=page.extract_text() #extracting text 
                print(f"Extracting text from file: {pdf}, page number: {page_number+1}")

                text = re.sub(r'\n+', '\n', text)  #remove multiple new lines
                text = re.sub(r'\s+', ' ', text) #remove multiple spaces

            
                chunks.append({'source':pdf,'page_number':page_number+1,"text":text.strip()}) #appending the chunk with meta data in the list

        with open(f'jsons/{pdf}.json','w',encoding='utf-8') as f: #writing the chunks into a json file
            json.dump({'chunks':chunks},f,indent=4) 

def create_embeddings(text_list): #function to create embeddings using BGE-M3 model by instalatling and running the BGE-M3 server locally
    response = requests.post('http://localhost:11434/api/embed',json={
        'model':"bge-m3",
        'input': text_list
    })
    embeddings=response.json()['embeddings'] #list of embeddings

    return(embeddings)

# Initialize ChromaDB client 
client = chromadb.PersistentClient(
        path="chroma_db",  # folder on disk
    )

def process_chunks_to_embeddings(): #function to process all the json files to create embeddings and store in ChromaDB

    jsons = os.listdir('jsons') #list of all json files inside jsons directory
    
    collection=client.create_collection(name='RAG-PDF') #create a collection named RAG-PDF
    
    for json_file in jsons: #iterate all json files 1 by 1

        with open (f"jsons/{json_file}",'r',encoding='utf-8') as file: #open the json file
            content=json.load(file) #load the content of json file

        print((f"Processing file: {json_file} with {len(content['chunks'])} chunks"))
        embeddings=create_embeddings([chunk['text'] for chunk in content['chunks']]) #create embeddings for all chunks in that json file

        for i,chunk in enumerate(content['chunks']): #iterate all chunks to add them to ChromaDB collection
            chunk['embedding']=embeddings[i] #add embedding to chunk dictionary


            # add chunk to ChromaDB collection, ID is combination of source file name and page number to make it unique
            collection.add( 
                documents=[chunk['text']], 
                embeddings=[embeddings[i]],
                metadatas=[{'source':chunk['source'],'page_number':chunk['page_number']}], 
                ids=[f"{chunk['source']}_{chunk['page_number']}" ]
            )


def process_query(): #function to process user query, retrieve relevant chunks from ChromaDB and get response from LLM

    query =input("Enter your query: ")
    print(f"Thinking on your query...")

    query_embedding=create_embeddings([query])[0] #create embedding for user query

    collection = client.get_collection(
    name="RAG-PDF"
    ) #get the collection named RAG-PDF

    #query the collection to get top 4 relevant chunks
    results = collection.query(
    query_embeddings=[query_embedding], # Chroma will embed this for you
    n_results=4 # how many results to return
    ) 

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    if not docs:
        print("No relevant documents found.")
        return

    docs_with_meta = list(zip(docs, metas))

    # call LLM to get response using the retrieved chunks
    get_response_from_llm(query, docs_with_meta)

    

def get_response_from_llm(query,documents_with_meta): #function to get response from LLM using OpenAI Responses API

    #Prompt template, PROMPT IS THE KEY FACTOR IN GETTING GOOD RESPONSE FROM LLM WITH THE RELEVANT CHUNKS INCLUDED
    prompt = f'''
            Your role is to answer the user queries related to the Data Science course. If a user asks something unrelated to the course just tell I'm Ultimate Data Science Assistant for the Data Science course and please Ask from the course related queries only.
            Here are the top 4 containing source file, page Number, Text  that have high cosine similarity to user query: {'|\n '.join([f"File Name: {meta['source']}, Page Number: {meta['page_number']}, Text: {doc}" for doc,meta in documents_with_meta])}
            --------------------------------------------------------
            User will ask query and you have to answer How much content is taught in which file at what page number and guide the user to go to that pdf.
    '''
   
    # ----------------- Api call to OpenAi responses API -----------------------

    req = requests.post("https://api.openai.com/v1/responses", 
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
                        },
                        json= {"model":"gpt-4.1-mini",
                        "instructions": prompt,
                        "temperature": 0.6,
                        "input": f"{query}",
                        "store":False,
    })
    
    if not req.ok:
        raise RuntimeError(f"OpenAI API error: {req.status_code} {req.text}")
    response=req.json()['output'][0]['content'][0]['text']

    #Storing the prompt and response in text files
    with open('prompt.txt','w',encoding='utf-8') as f:
        f.write(prompt)

    with open('response.txt','w',encoding='utf-8') as f:
        f.write(response)


task=int(input("Enter task: \n 0 for Extracting text from pdf and save as json in chunks. \n 1 for creating Embeddings of the extracted text chunks.\n 2 for entering your query and get relevant response.\n"))
if task==0:
    extract_txt_create_chunks()
elif task == 1:
    process_chunks_to_embeddings()
elif task==2:
    process_query()
else:
    print("Invalid Input")
      