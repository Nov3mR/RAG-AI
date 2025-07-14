from promptLLM import call_LLM
from getQueryPromts import *
import json
import re

def returnQuery(query: str):

    jsonPrompt = getJsonPrompt(query)
    newQueryPrompt = getNewPrompt(query)
    arithmeticPrompt = getArithmeticPrompt(query)

    jsonResponse, newQueryResponse = "", ""

    print("Calling LLM for JSON Prompt")
    jsonResponse = call_LLM(jsonPrompt)
    print("Calling LLM for New Query Prompt")
    newQueryResponse = call_LLM(newQueryPrompt)
    print("Calling LLM for Arithmetic Prompt")
    arithmeticResponse = call_LLM(arithmeticPrompt)

    cleaned = re.sub(r"^```(?:json)?\n?", "", jsonResponse.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\n?```$", "", cleaned.strip())
    cleaned = re.sub(r'//.*', '', cleaned)

    cleanedArithmeticResponse =  re.sub(r"^```(?:json)?\n?", "", arithmeticResponse.strip(), flags=re.IGNORECASE)
    cleanedArithmeticResponse = re.sub(r"\n?```$", "", cleanedArithmeticResponse.strip())
    cleanedArithmeticResponse = re.sub(r'//.*', '', cleanedArithmeticResponse)

    return cleaned, newQueryResponse, cleanedArithmeticResponse

if __name__ == "__main__":

    query = "How many invoices have not been paid at all in the supplier aging report"
    test1, test2, result = returnQuery(query)
    print(result)