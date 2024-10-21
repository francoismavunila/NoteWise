from langserve import RemoteRunnable

remote_chain = RemoteRunnable("http://localhost:8000/chain/")
res = remote_chain.invoke({"language": "italian", "text": "hi"})
print(res)