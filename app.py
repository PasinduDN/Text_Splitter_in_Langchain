# Corrected Code
from langchain_community.document_loaders import DirectoryLoader
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Provide the path to the DIRECTORY, not the file
loader = DirectoryLoader("data/text_data/", glob="**/*.txt") 

dataset = loader.load()

# for data in dataset:
#     print("-----------------------")
#     print(data.page_content)
#     print(data.metadata)

#Calculate the token count of each document in the dataset using a character level tokenizer
token_counter = [len(data.page_content) for data in dataset]
# print(token_counter)

#Tiktoken
tokenizer_model = tiktoken.encoding_for_model('gpt-3.5-turbo')
# print(tokenizer_model)

# get the encoding for the tokenizer 
tokenizer = tiktoken.get_encoding('cl100k_base')

#create a function to calculat the length of text in tokens using tiktoken
def tiktoken_len(text):
    tokens = tokenizer.encode(text)
    token_length = len(tokens)
    return token_length

#calculate the token count of each document in the dataset using tiktoken
token_counts = [tiktoken_len(data.page_content) for data in dataset]

# print(token_counts)

#Chunking the Text using RecursiveCharacterTextSplitter
#Initialize the text splitter with a chunk size of 150 and no overlap, using character length function
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 150,
    chunk_overlap = 0,
    length_function = len
)

# print(text_splitter)

#split the text of the second document in the dataset into chunk
chunks = text_splitter.split_text(dataset[0].page_content)

# print(chunks[1])
# print(len(chunks[1]))

# print("----------------------------------------------")

#Initialize the text splitter with a chunk size of 150 and no overlap, using character length function
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 150,
    chunk_overlap = 20,
    length_function = len
)

# print(text_splitter)

#split the text of the second document in the dataset into chunk
chunks = text_splitter.split_text(dataset[0].page_content)

# print(chunks[1])
# print(len(chunks[1]))


#Chunking the text using tiktoken length function
#Re initialize the text splitter with a chunk size of 150 an overlap of 20, using tiktoken length function

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 150,
    chunk_overlap = 20,
    length_function = tiktoken_len,
    separators=['\n\n', '\n', ' ','' ] 
)

#Split the text of the second document in the dataset into chunks
chunks = text_splitter.split_text(dataset[1].page_content)

print(len(chunks))

print("------------------------")

print(chunks[0])
print(tiktoken_len(chunks[0]))
