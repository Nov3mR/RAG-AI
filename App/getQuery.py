from promptLLM import call_LLM
from getQueryPromts import *
import json
import re

def returnNewQuery(query: str):

    jsonPrompt = getJsonPrompt(query)
    newQueryPrompt = getNewPrompt(query)

    jsonResponse, newQueryResponse = "", ""

    print("Calling LLM for JSON Prompt")
    jsonResponse = call_LLM(jsonPrompt)
    print(jsonResponse)
    # print("Calling LLM for New Query Prompt")
    # newQueryResponse = call_LLM(newQueryPrompt)
    print(newQueryResponse)

    cleaned = re.sub(r"^```(?:json)?\n?", "", jsonResponse.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\n?```$", "", cleaned.strip())
    cleaned = re.sub(r'//.*', '', cleaned)
    cleaned = re.sub(r",\s*(\}|\])", r"\1", cleaned)

    return cleaned, newQueryResponse

def returnArithmeticData(query: str) -> dict:
    arithmeticPrompt = getArithmeticPrompt(query)

    print("Calling LLM for Arithmetic Prompt")
    arithmeticResponse = call_LLM(arithmeticPrompt)
    print(arithmeticResponse)

    cleanedArithmeticResponse =  re.sub(r"^```(?:json)?\n?", "", arithmeticResponse.strip(), flags=re.IGNORECASE)
    cleanedArithmeticResponse = re.sub(r"\n?```$", "", cleanedArithmeticResponse.strip())
    cleanedArithmeticResponse = re.sub(r'//.*', '', cleanedArithmeticResponse)

    match = re.search(r"\{.*\}", cleanedArithmeticResponse, re.DOTALL)

    response = match.group(0) if match else cleanedArithmeticResponse

    json_str = re.sub(r",\s*(\}|\])", r"\1", response)

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        print("Error decoding JSON response")
        return {}

if __name__ == "__main__":

    query = "How many invoices are more than 180 days due and haven't been fully paid?"
    test1 = returnArithmeticData(query)
    print(test1)