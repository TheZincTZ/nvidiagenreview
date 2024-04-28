import pandas as pd
from langchain_community.llms.ollama import Ollama
from pandasai import SmartDataframe

data = pd.read_csv("cleaned_good_stuff.csv")
data.head()

llm = Ollama(model="llama3")

df = SmartDataframe(data, config={"llm": llm})

df.chat('distribution of text rating?')