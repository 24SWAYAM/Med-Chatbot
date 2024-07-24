from flask import Flask, render_template, jsonify, request
from src.helper import *
from langchain_community.vectorstores import Pinecone
import pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os
from accelerate import Accelerator
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.vectorstores import Pinecone as LangchainPinecone
# from langchain.chains import RetrievalQA
from pinecone import Pinecone
# from langchain_pinecone import Pinecone
app = Flask(__name__)

load_dotenv()


# print(PINECONE_API_KEY)

embeddings = download_hugging_face_embeddings()
index_name = "medical-chatbot"

pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY")
)
index = pc.Index(index_name)
index_name = "medical-chatbot"
loader = PyPDFDirectoryLoader("data")
extracted_data = loader.load()
text_chunks = text_split(extracted_data)
print("length of my chunk:", len(text_chunks))
# Create embeddings for your text chunks
embedded_texts = embeddings.embed_documents([t.page_content for t in text_chunks])

# Prepare vectors for upsert
vectors_to_upsert = []
for i, (chunk, embedding) in enumerate(zip(text_chunks, embedded_texts)):
    vector = {
        "id": f"chunk_{i}",
        "values": embedding,
        "metadata": {
            "text": chunk.page_content,
            # Add any other metadata you want to include
        }
    }
    vectors_to_upsert.append(vector)


# Split vectors into smaller batches
batch_size = 100  # You might need to adjust this
batches = chunk_list(vectors_to_upsert, batch_size)

# Upsert batches to Pinecone
for i, batch in enumerate(batches):
    try:
        index.upsert(
            vectors=batch,
            namespace="ns1"  # Replace with your desired namespace
        )
        print(f"Batch {i+1}/{len(batches)} upserted successfully")
    except Exception as e:
        print(f"Error upserting batch {i+1}: {str(e)}")
        # You might want to implement retry logic here

print("Upsert completed")
docsearch = LangchainPinecone(index, embeddings.embed_query, "text")

PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs={"prompt": PROMPT}

llm=CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':512,
                          'temperature':0.8})
accelerator = Accelerator()
llm = accelerator.prepare(llm)
qa=RetrievalQA.from_llm(
    llm=llm, 
    # chain_type="stuff", //
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True, 
    # chain_type_kwargs=chain_type_kwargs////////////////////////////)
)


@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)


