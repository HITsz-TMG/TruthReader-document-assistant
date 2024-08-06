import os
import re
import socket
import logging
from datetime import datetime
from typing import Optional
from collections import OrderedDict, defaultdict

from rouge import Rouge
from transformers import AutoTokenizer
from langdetect import detect_langs, DetectorFactory


def format_current_timestamp() -> str:
    """Format the current timestamp in seconds with dashes between each unit.

    Returns:
        str: The formatted timestamp as a string in the format 'yyyy-MM-dd-HH-mm-ss'.
    """
    now = datetime.now()
    formatted_timestamp = now.strftime('%Y-%m-%d-%H-%M-%S')
    return formatted_timestamp


def get_local_ip() -> str:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
        sock.close()
        return ip
    except Exception as e:
        logging.error("Geting local IP failed:", e)
        return None
    

def is_chinese_document(text: str) -> bool:
    DetectorFactory.seed = 0
    results = {r.lang: r.prob for r in detect_langs(text)}
    return results.get("zh-cn", 0) > 0.5
    

def extract_file_name(
        file_path: str, 
        tokenizer: AutoTokenizer, 
        max_len: int = 20
    ) -> str:
    file_name = os.path.basename(file_path)

    file_basic_name, file_extension = os.path.splitext(file_name)
    file_basic_name_ids = tokenizer(file_basic_name, add_special_tokens=False)

    if max_len is not None and len(file_basic_name_ids) > max_len:
        ellipsis_ids = tokenizer("...", add_special_tokens=False)
        file_name = tokenizer.decode(file_basic_name_ids[:max_len//2] + ellipsis_ids + file_basic_name_ids[-(max_len//2):]) + file_extension
    
    return file_name


def extract_citation(s: str) -> Optional[str | None]:
    if not isinstance(s, str):
        return None
    
    # pattern = re.compile(".*\n\n<small><font color=gray>🔗Reference(.*)</font></small>$", re.DOTALL)
    pattern = re.compile(""".*<small id="Reference"><font color=gray>(.*)</font></small>$""", re.DOTALL)
    matches = re.findall(pattern, s)
    if matches:
        citation = matches[0]
        return citation.strip()
    else:
        return None
    

def extract_citation_item(s: str) -> Optional[tuple | None]:
    pattern = r"\[(\d+)\]\s+(.+)\s+<sup>📃(\d+)</sup>"

    matches = re.findall(pattern, s.strip())
    if matches:
        item_id, title, page_id = matches[0]
        return item_id, title, page_id
    else:
        return None, None, None
    

def is_string_endswith_citation(s: str) -> bool:
    pattern = r'\[\d*$'
    match = re.search(pattern, s)
    if match:
        return True
    else:
        return False
    
    
def compute_sim(
        hyp: str, 
        ref: str, 
        tokenizer: AutoTokenizer
    ) -> float:
    rouge_func = Rouge(['rouge-1'])

    def cut_sentence(s):
        return " ".join([str(t).lower() for t in tokenizer.encode(s)])

    try:
        score = rouge_func.get_scores(cut_sentence(hyp), cut_sentence(ref))
        sim=  score[0]['rouge-1']['p']
    except Exception as e:
        logging.exception(e)
        sim = 0
    return round(sim, 2)


def wrap_answer_with_factual_score(
        text: str, 
        tokenizer: AutoTokenizer, 
        retrieved_documents: list
    ) -> str:
    pattern = r"((?:\[\d+\])+)"
    matches = re.search(pattern, text)
    if matches:
        match = matches.group()
        span_start = matches.span()[0]
        span_end = matches.span()[1]

        hyp = text[:span_start]

        citation_list = list(map(int, re.findall(r"\d+", match)))
        citation_list = [i for i in citation_list if i >= 1 and i <= len(retrieved_documents)]
        citation_documents = [retrieved_documents[i-1] for i in citation_list]
        ref = "\n".join([item.page_content for item in citation_documents])

        if len(hyp.strip()) > 0 and len(ref.strip()) > 0:
            score =  compute_sim(hyp, ref, tokenizer)

            suffix = wrap_answer_with_factual_score(text[span_end:], tokenizer, retrieved_documents)

            if score < 0.3:
                progress_class = "progress-1"
            elif score < 0.65:
                progress_class = "progress-2"
            else:
                progress_class = "progress-3"

            text = """{prefix}<sub><progress class="factual-score {progress_class}" value="{score}" max="1"></sub>{suffix}""".format(prefix=text[:span_end], progress_class=progress_class, score=score, suffix=suffix)
    return text
    

def reorder_reference(
        text: str, 
        num_reference: int
    ) -> tuple:
    """
    reorder citation label by their apperance in the response
    """
    pattern = r"\[(\d+)\]"

    matches = re.findall(pattern, text)
    matches = sorted(list(OrderedDict.fromkeys(matches)))

    cited_reference = [int(i) for i in matches if int(i) >= 1 and int(i) <= num_reference]
    uncited_reference = [i for i in range(1, num_reference+1, 1) if i not in cited_reference]
    reference_replace_dict = {old: new+1 for new, old in enumerate(cited_reference+uncited_reference)}
    return reference_replace_dict, cited_reference


def replace_bracket_references(
        text: str, 
        num_reference: int
    ) -> str:
    pattern = r"\[(\d+)\]"

    def replace_func(match):
        reference = match.group(1)
        if int(reference) > 0 and int(reference) <= num_reference:
            # return '<a class="citation-button" href="#reference-{0}">[{0}]</a>'.format(reference)
            return """<a class="citation-button" href="#reference-{0}">{0}</a>""".format(reference)
        else:
            return '[{0}]'.format(reference)

    result = re.sub(pattern, replace_func, text)
    return result


def restore_progress(text: str) -> str:
    pattern = """<sub><progress class="factual-score progress-\d" value="(.*?)" max="1"></sub>"""
    pattern = re.compile(pattern)

    def replace_func(match):
        value = match.group(1)
        return ''

    result = re.sub(pattern, replace_func, text)
    return result


def restore_bracket_references(text: str) -> str:
    """
    restore the hyperline to raw citation
    """
    # pattern = r'<a class="citation-button" href="#reference-(\d+)">\[(\d+)\]</a>'
    pattern = """<a class="citation-button" href="#reference-(\d+)">(\d+)</a>"""
    pattern = re.compile(pattern)

    def replace_func(match):
        reference = match.group(2)
        return '[{}]'.format(reference)

    result = re.sub(pattern, replace_func, text)
    return result


def genereate_retrieval_prompt(
        question: str, 
        history: list, 
        tokenizer: AutoTokenizer, 
        history_max_len: int = 500
    ) -> str:
    def cut_utterance(utterance):
        utterance_len = 128
        utterance_ids = tokenizer.encode(utterance, add_special_tokens=False)
        if len(utterance_ids) > utterance_len:
            utterance = tokenizer.decode(utterance_ids[:utterance_len])
        return utterance

    prompt = "# QUESTION: {question}\n\n{sep_token} # HISTORY:\n{history_string}"

    query = prompt.format(question=question, sep_token=tokenizer.sep_token, history_string="None")
    for i in range(0, len(history)):
        if len(history[i:]) > 0:
            history_list = ["A: {}\nB: {}".format(cut_utterance(history[j][0]), cut_utterance(history[j][1]))
                            for j in range(i, len(history))]
            history_list.reverse()
            history_string = "||".join(history_list)
        else:
            history_string = "None"
        query = prompt.format(question=question, sep_token=tokenizer.sep_token, history_string=history_string)

        if len(tokenizer.encode(query)) < history_max_len:
            break
    
    # logging.info("-"*50 + " retrieval query")
    # logging.info(query)
    return query


def generate_qa_prompt(
        question: str, 
        documents: list, 
        tokenizer: AutoTokenizer, 
        max_context_len: int = 3000
    ) -> str:
    documents_list = []
    for i, item in enumerate(documents):
        title = item.metadata.get("title", "")

        text = str(item.page_content).strip()
        if text.startswith(title):
            text = text[len(title):].strip()
        
        title = tokenizer.decode(tokenizer.encode(title, add_special_tokens=False)[:100])
        text = tokenizer.decode(tokenizer.encode(text, add_special_tokens=False)[:max_context_len])

        document = f"""## 文档[{i+1}]\t{title}-{item.metadata["page_id"]}\n{text}"""
        documents_list.append(document)
    
    documents_string = "\n\n".join(documents_list)
    for i in range(len(documents_list), 0, -1):
        documents_string = "\n\n".join(documents_list[:i])
        if len(tokenizer.encode(documents_string, add_special_tokens=False)) < max_context_len:
            break
    
    system = '''请基于给定的文档，生成问题的答案。如果文档中没有包含答案的信息，请回复抱歉并给出理由。'''
    prompt = "{system}\n\n# DOCUMENTS:\n{documents}\n\n# QUESTION: {Question}\n\n# ANSWER: "

    qa_prompt = prompt.format_map({
        "system": system,
        "documents": documents_string,
        "Question": question
    })
    logging.info("-"*50 + " generaiton instruction")
    logging.info(qa_prompt)
    return qa_prompt


def sort_documents_by_doc_page(documents: list) -> list:
    documents_dict = defaultdict(list, OrderedDict())
    for item in documents:
        documents_dict[item.metadata["title"]].append(item)

    for title in documents_dict:
        documents_dict[title] = sorted(documents_dict[title], key=lambda item: item.metadata["page_id"])
    
    sorted_documents = [item for title in documents_dict for item in documents_dict[title]]
    return sorted_documents
