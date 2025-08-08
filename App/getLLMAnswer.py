from buildQuery import *
from promptLLM import *



def returnLLMAnswer(query, context=None, meta=None, arithmeticResult=None, arithmeticRow=None, newContext=None, newMeta=None, llmMode="openai", history_context=""):
    """
    This function is used to get the answer from the LLM for a given file and question. MAKES THE PROMPT
    """

    if arithmeticResult and isinstance(arithmeticResult, float) and (arithmeticRow == "" or arithmeticRow is None):
        context = ""
        if newMeta is not None and newContext is not None:
            context = context if context is not None else "" + "\n" + newContext
            meta = meta if meta is not None else "" + "\n" + newMeta

        prompt = history_context + returnArithmeticQuery(context=context, meta=meta, query=query, arithmetic=arithmeticResult)
        print("Calling LLM for Arithmetic Prompt")
        llmAnswer = call_LLM(prompt=prompt, mode=llmMode)
        answer = llmAnswer
    elif arithmeticResult and (arithmeticRow == "" or arithmeticRow is None):
        context = ""
        if newMeta is not None and newContext is not None:
            context = context if context is not None else "" + "\n" + newContext
            meta = meta if meta is not None else "" + "\n" + newMeta
        prompt = history_context + returnArithmeticQuery(context=context, meta=meta, query=query, arithmetic=arithmeticResult)
        print("Calling LLM for Arithmetic Prompt")
        llmAnswer = call_LLM(prompt=prompt, mode=llmMode)
        answer = llmAnswer
    elif arithmeticRow != "" and arithmeticRow is not None:
        if isinstance(arithmeticRow, str):
            arithmeticRow = arithmeticRow.strip()
            context = "\n".join(arithmeticRow.split())
        elif isinstance(arithmeticRow, list):
            arithmeticRow = [row.strip() for row in arithmeticRow if row.strip()]
            context = "\n".join(arithmeticRow)
        meta = meta
        query = query

        if newMeta is not None and newContext is not None:
            context = context if context is not None else "" + "\n" + newContext
            meta = meta if meta is not None else "" + "\n" + newMeta

        prompt = history_context + returnArithmeticQuery(context=context, meta=meta, query=query, arithmetic=arithmeticResult)
        print("Calling LLM for Arithmetic Prompt")
        llmAnswer = call_LLM(prompt=prompt, mode=llmMode)
        answer = llmAnswer
    else:
        if newMeta is not None and newContext is not None:
            context = context if context is not None else "" + "\n" + newContext
            meta = meta if meta is not None else "" + "\n" + newMeta
        prompt = history_context + returnQuery(context=context, meta=meta, query=query)
        print("Calling LLM")
        answer = call_LLM(prompt=prompt, mode=llmMode)

    return answer
