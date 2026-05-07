import re

text = "Transformers use attention. They process tokens in parallel. They use KVP to maintain attention"

sentence_endings = re.compile(r'(?<=[.!?])\s+')

sentences = sentence_endings.split(text)

# print(sentences)
chunk=[]
current_chunk =[]
chunk_size=80

current_length = 0


for sentense in sentences:
    if current_length + len(sentense)<chunk_size:
        current_chunk.append(sentense)
        current_length = len(" ".join(current_chunk))    
        print(current_chunk)   
    else :
        chunk.append(" ".join(current_chunk))
        current_chunk =current_chunk[-1:]
        current_chunk.append(sentense)
        current_length = len(" ".join(current_chunk))
        print(chunk)
if current_chunk:
    chunk.append(" ".join(current_chunk))
print(chunk)