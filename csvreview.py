from langchain.document_loaders import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

loader = CSVLoader("cleaned_good_stuff.csv")
documents = loader.load()
embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-mpnet-base-v2')

chroma_db = Chroma.from_documents(
    documents, embeddings, persist_directory="./chroma_db"
)
chroma_db.persist()

llm = Ollama(model="llama3")

prompt_template = PromptTemplate(
    input_variables=["context"],
    template="Given this context: {context}, please directly answer the question: {question}.",
)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=chroma_db.as_retriever(),
    chain_type_kwargs={"prompt": prompt_template},
)

question = "what type of weather will the product work in best?"
result = qa_chain({"query": question})
print(result)