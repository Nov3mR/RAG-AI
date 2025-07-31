from buildQuery import *
from promptLLM import *

def returnLLMAnswer(query, context=None, meta=None, arithmeticResult=None, arithmeticRow=None, newContext=None, newMeta=None):
    """
    This function is used to get the answer from the LLM for a given file and question. MAKES THE PROMPT
    """
    if arithmeticResult and isinstance(arithmeticResult, float) and (arithmeticRow == "" or arithmeticRow is None):
        answer = f"The result is: {arithmeticResult:,.2f}"
    elif arithmeticResult and (arithmeticRow == "" or arithmeticRow is None):
        answer = f"The result is: {arithmeticResult}"
    elif arithmeticRow != "" and arithmeticRow is not None:
        context = "\n".join(arithmeticRow.split())
        meta = meta
        query = query
        prompt = returnArithmeticQuery(context=context, meta=meta, query=query, arithmetic=arithmeticResult)
        print("Calling LLM for Arithmetic Prompt")
        answer = f"The result is: {arithmeticResult}"
        llmAnswer = call_LLM(prompt=prompt)
        answer += f"\n{llmAnswer}"
    else:
        if newMeta is not None and newContext is not None:
            context = context if context is not None else "" + "\n" + newContext
            meta = meta if meta is not None else "" + "\n" + newMeta
        prompt = returnQuery(context=context, meta=meta, query=query)
        print("Calling LLM")
        answer = call_LLM(prompt=prompt)

    return answer
