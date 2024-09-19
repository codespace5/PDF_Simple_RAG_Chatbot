import os
import warnings
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore

warnings.filterwarnings("ignore")



chat_history = []

if __name__ == "__main__":
    os.environ['PINECONE_API_KEY'] = ''
    os.environ['OPENAI_API_KEY'] = ''
    embeddings = OpenAIEmbeddings(openai_api_key="")
    vectorstore = PineconeVectorStore(
        index_name="bdobot", embedding=embeddings
    )

    chat = ChatOpenAI(verbose=True, temperature=0, model_name="gpt-4o")

    qa = RetrievalQA.from_chain_type(
        llm=chat, chain_type="stuff", retriever=vectorstore.as_retriever()
    )

    res = qa.invoke("What are the applications of generative AI according the the paper? Please number each application.")
    print(res) 

    res = qa.invoke("Can you please elaborate more on application number 2?")
    print(res)
    res = qa.invoke("what is an apple?")
    print(res)
