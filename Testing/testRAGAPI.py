import requests

def ask_rag(query):
    url = "http://localhost:8000/query"
    response = requests.post(url, json={"query": query})
    return response.json()

if __name__ == "__main__":
    q = input("Enter your tax query: ")
    result = ask_rag(q)
    print("\nAnswer:\n", result["answer"])
