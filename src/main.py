import os
import re
import copy
import logging
import random
import threading
import argparse
import tempfile
from datetime import datetime
from pathlib import Path
from collections import OrderedDict, defaultdict
from typing import List

import torch
import gradio as gr
from gradio.events import on
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from openai import OpenAI

from parser import load_file, load_url, split_chunks, PDFProcesser
from config import max_doc_num, max_page_num, max_context_len, max_model_len, retrieved_doc_num
from utils import (
    format_current_timestamp,
    get_local_ip,
    is_chinese_document,
    extract_file_name,
    is_string_endswith_citation,
    wrap_answer_with_factual_score,
    reorder_reference,
    replace_bracket_references,
    restore_bracket_references,
    restore_progress,
    generate_qa_prompt, 
    genereate_retrieval_prompt, 
    extract_citation, 
    extract_citation_item, 
    sort_documents_by_doc_page,
)
from example import example


SPLIT_LINE_PAGE = """\n<hr class="hr-edge-weak">\n"""
SPLIT_LINE_DOC = """\n<hr class="hr-double-arrow">\n"""


def summarize_single_doc(
        title: str, 
        document: str, 
        url: str, 
        input_max_len: int = 3000, 
        temperature: float = 0.75, 
        top_p: float = 0.75, 
        max_new_tokens: int = 1024
    ) -> Document:
    if len(document) == 0:
        return []
    
    model_selection = "Qwen-14B"
    chat_tokenizer: AutoTokenizer = chat_tokenizer_dict[model_selection]
    
    title = chat_tokenizer.decode(chat_tokenizer.encode(title, add_special_tokens=False)[:100])
    document = chat_tokenizer.decode(chat_tokenizer.encode(document, add_special_tokens=False)[:input_max_len])
    document_string = f'## 文档[1]\t{title}\n{document}'

    system = '''请基于给定的文档，生成问题的答案。如果文档中没有包含答案的信息，请回复抱歉并给出理由。'''
    if is_chinese_document(document[:4096]):
        question = "请分点概括这篇文章的主要内容"
        
        summary_prefix = "# 文章摘要\n\n"
    else:
        question = "Please systematically summarize the main content of this document."
        summary_prefix = "# Document Abstract\n\n"
    
    prompt = "{system}\n\n# DOCUMENTS:\n{documents}\n\n# QUESTION: {Question}\n\n# ANSWER: "
    instruction = prompt.format(system=system, documents=document_string, Question=question)

    messages = [{"role": "system", "content": "You are a helpful assistant."}] + [{"role": "user", "content": instruction}]
    
    port = 8001

    client = OpenAI(
        api_key="EMPTY",
        base_url="{chat_model_server}:{port}/v1".format(chat_model_server=chat_model_server, port=port),
    )
    response = client.chat.completions.create(
        model="/vllm/EMPTY",
        messages=messages,
        stream=False,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        timeout=20,
    ).choices[0].message.content

    summary = summary_prefix + response.replace("[1]", "")
    metadata = {"url": url, "page_id": 0, "title": title}

    return [Document(page_content=summary, metadata=metadata)]


def upload_file_fn(
        file_list: List[str], 
        page_size: int = 400, 
        page_overlap_slider: int = 0, 
        pre_summary: bool = True, 
        use_pdf_processor: bool = True,
        progress: gr.Progress = gr.Progress(track_tqdm=True)
    ):
    faiss_db = None
    doc_text_dict = {}
    for file in progress.tqdm(file_list[:max_doc_num]):
        file_path = file.name
        logging.info("Upload file to {}.".format(file_path))
        file_name = extract_file_name(file_path, embedding_tokenizer, max_len=None)
        
        document_text = load_file(file_path, pdf_processer) if use_pdf_processor else load_file(file_path, None)

        summary = summarize_single_doc(title=file_name, document=document_text, url=file_path) if pre_summary else []
        doc_pages = summary + split_chunks(document_text, chunk_size=page_size, chunk_overlap=page_overlap_slider, url=file_path, title=file_name, tokenizer=embedding_tokenizer)[:max_page_num]

        if len(doc_pages) == 0:
            continue

        if faiss_db is None:
            faiss_db = FAISS.from_documents(doc_pages, embedding=embedding_model, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
        else:
            faiss_db.add_documents(doc_pages)

        doc_text_dict[file_name] = doc_pages
    
    file_name_list = list(doc_text_dict.keys())
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
    if len(file_name_list) == 0:
        return gr.Row(), gr.Accordion(label="Preprocessing Parameter", open=True), gr.Tabs(), gr.Dropdown(), gr.Textbox(), gr.Button(), gr.Button(), gr.Button(), gr.Button(), None, None, None
    
    return gr.update(visible=False), gr.Accordion(label="Preprocessing Parameter", open=False), gr.Tabs(visible=True, selected="tab-doc-cont"), gr.update(choices=file_name_list, value=file_name_list[-1]), gr.update(placeholder="Input Question", interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), doc_text_dict, doc_pages, faiss_db


def clear_file_fn():
    return gr.update(visible=True), gr.Accordion(label="Preprocessing Parameter", open=True), gr.Tabs(visible=False, selected="tab-doc-cont"), gr.update(value=None), gr.update(choices=None, value=None), gr.update(interactive=False, placeholder="Please upload files first.", value=None), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), [], [], None, None, None, None


def url_load_fn(
        url_box: str, 
        page_size: int = 400, 
        page_overlap_slider: int = 0, 
        pre_summary: bool = True, 
        use_pdf_processor: bool = True, 
        progress: gr.Progress = gr.Progress(track_tqdm=True)
    ):
    urls = re.split("[,，\n]", url_box)
    urls = [url.strip() for url in urls]

    faiss_db = None
    doc_text_dict = {}
    for doc_id, url in enumerate(progress.tqdm(urls[:max_doc_num])):
        logging.info("Upload url from {}.".format(url))

        document_text, title = load_url(url, pdf_processer) if use_pdf_processor else load_url(url, None)

        if len(title) == 0:
            title = "Untitled-{}".format(doc_id+1)

        summary = summarize_single_doc(title=title, document=document_text, url=url) if pre_summary else []
        doc_pages = summary + split_chunks(document_text, chunk_size=page_size, chunk_overlap=page_overlap_slider, url=url, title=title, tokenizer=embedding_tokenizer)[:max_page_num]

        if len(doc_pages) == 0:
            continue

        if faiss_db is None:
            faiss_db = FAISS.from_documents(doc_pages, embedding=embedding_model, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
        else:
            faiss_db.add_documents(doc_pages)

        doc_text_dict[title] = doc_pages
    
    file_name_list = list(doc_text_dict.keys())
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
    if len(file_name_list) == 0:
        return gr.Row(), gr.Accordion(label="Preprocessing Parameter", open=True), gr.Tabs(), gr.Dropdown(), gr.Textbox(), gr.Button(), gr.Button(), gr.Button(), gr.Button(), None, None, None, gr.File()
    
    return gr.update(visible=False), gr.Accordion(label="Preprocessing Parameter", open=False), gr.Tabs(visible=True, selected="tab-doc-cont"), gr.update(choices=file_name_list, value=file_name_list[-1]), gr.update(placeholder="Input Question", interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), doc_text_dict, doc_pages, faiss_db, urls


def liked_feedback_collection_fn(
        data: gr.LikeData, 
        chatbot: gr.Chatbot
    ):
    history = chatbot[: data.index[0]+1]
    clean_history = [[history_pair[0], history_pair[1].split("""<small id="Reference"><font color=gray>""")[0].strip()] for history_pair in history]
    compact_clean_history = [h for history_pair in clean_history for h in history_pair]
    
    liked = int(data.liked)

    log_dic = {
        "feedback": {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "question": compact_clean_history[-2],
            "history": compact_clean_history[:-2],
            "answer": compact_clean_history[-1],
            "liked": liked,
        }
    }
    logging.info(log_dic)


def chat_listen_fn(chatbot_state):
    if chatbot_state is None or len(chatbot_state) == 0:
        return gr.Textbox()
    if chatbot_state[-1][1] is None or not chatbot_state[-1][1].endswith("</small>"):
        return gr.Textbox()
    citation = extract_citation(chatbot_state[-1][1])
    if citation is None or len(citation) == 0:
        return gr.Textbox()

    return gr.update(value=chatbot_state[-1][1])


def display_attribution_fn(
        fake_response: str, 
        doc_text_dict: dict
    ):
    # prefix = """<mark style="color: #0700b8;background: transparent;">"""
    # suffix = """</mark>"""
    prefix = "<strong>"
    suffix = "</strong>"

    citation = extract_citation(fake_response)
    raw_response = fake_response.split("""<small id="Reference"><font color=gray>""")[0]

    attribution_dict = defaultdict(list, OrderedDict())
    for citation_item in citation.split("</a>"):
        item_id, title, page_id = extract_citation_item(citation_item)
        if title is not None and page_id is not None:
            doc_pages = doc_text_dict.get(title)
            doc_pages_dict = {doc.metadata["page_id"]: doc for doc in doc_pages}
            if doc_pages is not None:
                doc_page_ = doc_pages_dict.get(int(page_id))
                if doc_page_ is not None:
                    page_content = doc_page_.page_content[len(doc_page_.metadata["title"]):].strip() if doc_page_.page_content.startswith(doc_page_.metadata["title"]) else doc_page_.page_content

                    if ">{}</a>".format(item_id) in raw_response:
                        page_content = prefix + page_content + suffix
                    attribution_dict[title].append("""<article id="reference-{i}"><h5>📃{page_id}</h5>{page_content}</article>""".format(i=item_id, page_id=page_id, page_content=page_content))

    doc_pages_list = ["""<h3 style="text-align: center;">{title}</h3>\n{page_content}""".format(title=title, page_content=SPLIT_LINE_PAGE.join(pages))
        for title, pages in attribution_dict.items()]

    doc_text = SPLIT_LINE_DOC.join(doc_pages_list)
    
    return gr.update(value=doc_text), gr.Tabs(selected="tab-attr-chunk")


def doc_select_fn(
        doc_selection_dropdown: str, 
        doc_text_dict: dict
    ):
    if doc_text_dict is None:
        return gr.update(value=None)
    else:
        doc_pages = doc_text_dict.get(doc_selection_dropdown, None)
        doc_text = SPLIT_LINE_PAGE.join(["""<article><h5>📃{page_id}</h5>{page_content}</article>""".format(
            page_id=doc_page_.metadata["page_id"],
            page_content=doc_page_.page_content[len(doc_page_.metadata["title"]):].strip() if doc_page_.page_content.startswith(doc_page_.metadata["title"]) else doc_page_.page_content
            ) for i, doc_page_ in enumerate(doc_pages)])
        return gr.update(value=doc_text)


def dochelper_chat_fn(
    question: str,
    history: list,
    model_selection: str = "Qwen-14B",
    temperature: float = 0.5,
    top_p: float = 1.0,
    top_k: int = 50,
    max_new_tokens: int = 1024,
    repetition_penalty: float = 1.1,
    no_repeat_ngram_size: int = 0,
    do_sample: bool = True,
    faiss_db: FAISS = None,
):
    if model_selection not in chat_tokenizer_dict:
        model_selection = "Qwen-14B"
    chat_tokenizer: AutoTokenizer = chat_tokenizer_dict[model_selection]
    chat_model: AutoModelForCausalLM = chat_model_dict[model_selection]

    generation_config = {
        'max_new_tokens': min(max_new_tokens, 1024),
        'temperature': min(max(0.01, temperature), 1.2),
        'top_p': min(max(0.01, top_p), 1.0),
        'top_k': min(max(0, top_k), 64),
        'num_beams': 1,
        'do_sample': do_sample,
        'repetition_penalty': min(max(0, repetition_penalty), 1.2),
        'no_repeat_ngram_size': min(max(0, no_repeat_ngram_size), 20),
        'min_new_tokens': 2,
    }

    # preprocess to remove the citation
    for h_pair in history:
        h_pair[1] = re.sub(r'href="#reference-(\d+)"', '', h_pair[1])
    history = copy.deepcopy(history)
    for h_pair in history:
        h_pair[1] = restore_bracket_references(restore_progress(h_pair[1].split("""<small id="Reference"><font color=gray>""")[0]))

    # Step2: retrieve documents
    retrieval_instruction = genereate_retrieval_prompt(question, history, embedding_tokenizer, history_max_len=500)
    
    retrieved_documents_with_score = faiss_db.similarity_search_with_score(retrieval_instruction, k=200)[:retrieved_doc_num]
    retrieved_documents = [item[0] for item in retrieved_documents_with_score]
    retrieved_documents = sort_documents_by_doc_page(retrieved_documents)

    # Step3: genereate response
    qa_instruction = generate_qa_prompt(question, retrieved_documents, chat_tokenizer, max_context_len=max_context_len)

    messages = []
    for user_content, assistant_content in history:
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": assistant_content})
    messages = [{"role": "system", "content": "You are a helpful assistant."}] + messages + [{"role": "user", "content": qa_instruction}]

    if "LyChee-6B" == model_selection:
        input_ids = chat_tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            max_length=max_model_len-min(max_new_tokens, 1024),
            truncation=True,
        ).to(chat_model.device)

        streamer = TextIteratorStreamer(chat_tokenizer, skip_prompt=True, skip_special_tokens=True)

        generation_kwargs = dict(input_ids=input_ids, streamer=streamer)
        generation_kwargs.update(generation_config)
        thread = threading.Thread(target=chat_model.generate, kwargs=generation_kwargs)
        thread.start()

        generated_text = ""
        for new_text in streamer:
            generated_text += new_text
            if not is_string_endswith_citation(generated_text):
                yield replace_bracket_references(generated_text, num_reference=len(retrieved_documents))
        
    else:
        port_dict = {"Baichuan-13B": 8000, "Qwen-14B": 8001, "Mixtral-7B*2": 8002,}
        port = port_dict.get(model_selection, 8001)

        client = OpenAI(
            api_key="EMPTY",
            base_url="{chat_model_server}:{port}/v1".format(chat_model_server=chat_model_server, port=port),
        )
        streamer = client.chat.completions.create(
            model="/vllm/EMPTY",
            messages=messages,
            stream=True,
            temperature=temperature if do_sample else 0,
            top_p=top_p,
            max_tokens=max_new_tokens,
            timeout=20,
        )

        generated_text = ""
        for chunk in streamer:
            new_text = chunk.choices[0].delta.content
            if new_text is not None:
                generated_text += new_text
                if not is_string_endswith_citation(generated_text):
                    yield replace_bracket_references(generated_text, num_reference=len(retrieved_documents))


    # Step4: Post-process reponse for reference
    reference_replace_dict, cited_reference = reorder_reference(generated_text, num_reference=len(retrieved_documents))
    
    def replace_reference(match):
        old = int(match.group(1))
        new = reference_replace_dict.get(old, old)
        return f"[{new}]"
    generated_text = re.sub(r'\[(\d)\]', replace_reference, generated_text)
    
    reference_list = ["""{br}<a class="{elem_class}" href="#reference-{i}">[{i}] {title} <sup>📃{page_id}</sup></a>""".format(
        elem_class="hide" if i+1 not in cited_reference else "visiable",
        br="" if i+1 not in cited_reference else "\n",
        i=reference_replace_dict[i+1], 
        title=item.metadata["title"], 
        page_id=item.metadata["page_id"]) 
        for i, item in enumerate(retrieved_documents)]
    
    reference_head = "\n\n🔗Reference" if len(cited_reference) > 0 else ""
    reference_string = "".join(reference_list)
    citation_suffix = """<small id="Reference"><font color=gray>{reference_head}{reference_string}</font></small>""".format(reference_head=reference_head, reference_string=reference_string)
    generated_text = replace_bracket_references(wrap_answer_with_factual_score(generated_text, chat_tokenizer, retrieved_documents), num_reference=len(retrieved_documents)) + citation_suffix
    # generated_text = replace_bracket_references(generated_text, num_reference=len(retrieved_documents)) + citation_suffix

    logging.debug("Response:\n{}".format(generated_text))
    logging.debug("-"*100)
    yield generated_text

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def init(
        embed_model_path: str = None,
        chat_model_path: str = None,
        chat_model_server_address: str = None,
        orc_model_path: str = None,
        log_dir: str = None,
        tmp_folder: str = None,
    ):
    random.seed(1234)

    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(log_dir, "{}.log".format(format_current_timestamp()))
        logging.basicConfig(filename=log_file_path, level=logging.INFO, format='[%(levelname)s] [%(asctime)s] %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='[%(levelname)s] [%(asctime)s] %(message)s')
    

    if tmp_folder is not None:
        os.makedirs(tmp_folder, exist_ok=True)
        os.environ['GRADIO_TEMP_DIR'] = tmp_folder
        tempfile.tempdir = tmp_folder

    global device
    global pdf_processer
    global embedding_tokenizer
    global embedding_model
    global chat_tokenizer_dict
    global chat_model_dict
    global chat_model_server

    chat_model_server = chat_model_server_address
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    embedding_model = HuggingFaceBgeEmbeddings(
        model_name=embed_model_path,
        model_kwargs={"device": device}, 
        encode_kwargs={"normalize_embeddings": True, "batch_size": 4, "show_progress_bar": True}
    )
    embedding_tokenizer = embedding_model.client.tokenizer

    pdf_processer = PDFProcesser(orc_model_path)

    chat_tokenizer_dict = {}
    chat_model_dict = {}
    for model_path in chat_model_path.split(";"):
        if "lychee" in model_path.lower():
            model_name = "LyChee-6B"
            chat_model_dict[model_name] = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True, max_memory={0: "28GB",})
        elif "baichuan" in model_path.lower():
            model_name = "Baichuan-13B"
            chat_model_dict[model_name] = None
        elif "qwen" in model_path.lower():
            model_name = "Qwen-14B"
            chat_model_dict[model_name] = None
        elif "mixtral" in model_path.lower():
            model_name = "Mixtral-7B*2"
            chat_model_dict[model_name] = None
        else:
            raise NotImplementedError("Could not parser the model type from {}.".format(model_path))
        
        chat_tokenizer_dict[model_name] = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        chat_tokenizer_dict[model_name].truncation_side='left'
        

def main(
        embed_model_path: str = None,
        chat_model_path: str = None,
        chat_model_server: str = None,
        orc_model_path: str = None,
        log_dir: str = None,
        tmp_folder: str = None,
        port: int = 8080,
    ):
    init(embed_model_path, chat_model_path, chat_model_server, orc_model_path, log_dir, tmp_folder)

    with open(Path(__file__).parent / "resources/head.html") as html_file:
        head = html_file.read().strip()

    with gr.Blocks(theme=gr.themes.Soft(font="sans-serif").set(background_fill_primary="linear-gradient(90deg, #e3ffe7 0%, #d9e7ff 100%)", background_fill_primary_dark="linear-gradient(90deg, #4b6cb7 0%, #182848 100%)",),
                   head=head, 
                   css=Path(__file__).parent / "resources/styles.css", 
                   title="TruthReader", 
                   fill_height=True, 
                   analytics_enabled=False) as demo:
        doc_text_dict = gr.State()
        doc_pages = gr.State()
        faiss_db = gr.State()
        fake_response_textbox = gr.Textbox(label="fake_response_textbox", visible=False)

        with gr.Row(equal_height=False, elem_classes="row"):
            with gr.Column(min_width=600):
                with gr.Accordion(label="Preprocessing Parameter", open=False) as preprocessing_accordion:
                    page_size_slider = gr.components.Slider(minimum=200, maximum=750, step=10, value=750, label="Chunk Size", scale=1)
                    page_overlap_slider = gr.components.Slider(minimum=0, maximum=200, step=10, value=0, label="Chunk Overlap", scale=1)
                    summary_checkbox = gr.Checkbox(value=True, label="Document Pre-summarization  (cost more time)", scale=1)
                    pdf_processor_checkbox = gr.Checkbox(value=False, label="PDF VisionProcessor  (cost more time)", scale=1)

                doc_files_box = gr.File(label="Upload Documents", file_types=[".docx", ".md", ".pdf", ".txt"], file_count="multiple", height=90, scale=1)

                with gr.Group():
                    with gr.Row(equal_height=True) as url_row:
                        url_box = gr.Textbox(
                            scale=4,
                            max_lines=1,
                            show_label=False,
                            placeholder="Input URLs (multiple URLs separated by commas ',' )",
                            container=False,
                        )
                        url_botton = gr.Button("⚙️Fetch Web", size="sm", scale=1)

                with gr.Tabs(visible=False) as tabs:
                    with gr.Tab("Document Content", elem_classes="tab", id="tab-doc-cont") as doc_content_tab:
                        with gr.Accordion(open=True):
                            doc_selection_dropdown = gr.Dropdown(label=None, show_label=False, container=True, interactive=True, scale=1)
                            doc_display_html = gr.HTML(label="Document Content", elem_classes="html-text")

                    with gr.Tab("Attribution Chunks", elem_classes="tab", id="tab-attr-chunk") as attr_chunks_tab:
                        with gr.Accordion(open=True):
                            attribution_chunks_html = gr.HTML(label="Attribution Chunks", elem_classes="html-text")

            with gr.Column(min_width=600):
                chat_interface = gr.ChatInterface(
                    dochelper_chat_fn,
                    chatbot=gr.Chatbot(scale=1, label="chatbot", show_label=False, show_copy_button=True, likeable=True, layout="panel", render=False),
                    textbox=gr.Textbox(placeholder="Please upload files first.", container=False, scale=7, render=False, max_lines=4, interactive=False, value=None),
                    css=Path(__file__).parent / "themes/styles.css",
                    title="📚TruthReader",
                    theme="soft",
                    fill_height=True,
                    autofocus=False,
                    submit_btn=gr.Button("▶️  Submit", variant="primary", scale=1, min_width=150, interactive=False, render=False),
                    stop_btn=gr.Button("⏹  Stop", variant="stop", visible=False, scale=1, min_width=150, render=False),
                    retry_btn=gr.Button("🔄  Retry", variant="secondary", interactive=False, render=False, scale=1, size="sm"),
                    undo_btn=gr.Button("↩️  Undo", variant="secondary", interactive=False, render=False, scale=1,  size="sm"),
                    clear_btn=gr.Button("🗑️  Clear", variant="secondary", interactive=False, render=False, scale=1,  size="sm"),

                    additional_inputs=[
                        gr.Radio(["LyChee-6B", "Mixtral-7B*2", "Qwen-14B"], value="Qwen-14B", label="Model Selection", render=True),
                        gr.components.Slider(
                            minimum=0.01, maximum=1.20, step=0.01, value=0.8, label="Temperature", render=False,
                        ),
                        gr.components.Slider(
                            minimum=0.01, maximum=1.0, step=0.01, value=0.8, label="Top P", render=False,
                        ),
                        gr.components.Slider(
                            minimum=1, maximum=64, step=1, value=50, label="Top K", render=False,
                        ),
                        gr.components.Slider(
                            minimum=1, maximum=600, step=4, value=600, label="Max New Tokens", render=False,
                        ),
                        gr.components.Slider(
                            minimum=0, maximum=1.1, step=0.01, value=1.02, label="Repetition Penalty", render=False, ),
                        gr.components.Slider(
                            minimum=0, maximum=20, step=1, value=0, label="No Repeat Ngram", render=False, ),
                        gr.Checkbox(
                            value=True, label="Sampling", render=False,
                        ),
                        faiss_db,
                    ],
                    additional_inputs_accordion=gr.Accordion(label="Generation Parameter", open=False, render=False),
                    concurrency_limit=1,
                )

                examples = gr.Examples(example, 
                    inputs=[url_box, chat_interface.textbox]
                    )

            submit_triggers = [url_box.submit, url_botton.click]
            submit_event = on(
                submit_triggers,
                url_load_fn,
                [url_box, page_size_slider, page_overlap_slider, summary_checkbox, pdf_processor_checkbox], 
                [url_row, preprocessing_accordion, tabs, doc_selection_dropdown, chat_interface.textbox, chat_interface.submit_btn, chat_interface.retry_btn, chat_interface.undo_btn, chat_interface.clear_btn, doc_text_dict, doc_pages, faiss_db, doc_files_box],
                queue=True,
            ).then(
                doc_select_fn,
                [doc_selection_dropdown, doc_text_dict], 
                doc_display_html
            )

            doc_files_box.upload(
                upload_file_fn, 
                [doc_files_box, page_size_slider, page_overlap_slider, summary_checkbox, pdf_processor_checkbox], 
                [url_row, preprocessing_accordion, tabs, doc_selection_dropdown, chat_interface.textbox, chat_interface.submit_btn, chat_interface.retry_btn, chat_interface.undo_btn, chat_interface.clear_btn, doc_text_dict, doc_pages, faiss_db],
                queue=True,
                trigger_mode="once"
            ).then(
                doc_select_fn,
                [doc_selection_dropdown, doc_text_dict], 
                doc_display_html
            )
            
            doc_files_box.clear(
                clear_file_fn, 
                None, 
                [url_row, preprocessing_accordion, tabs, doc_display_html, doc_selection_dropdown, chat_interface.textbox, chat_interface.submit_btn, chat_interface.retry_btn, chat_interface.undo_btn, chat_interface.clear_btn, chat_interface.chatbot, chat_interface.chatbot_state, chat_interface.saved_input, doc_text_dict, doc_pages, faiss_db]
            )

            doc_selection_dropdown.change(doc_select_fn, [doc_selection_dropdown, doc_text_dict], doc_display_html)

            chat_interface.chatbot.like(liked_feedback_collection_fn, chat_interface.chatbot, None)
            chat_interface.chatbot.change(chat_listen_fn, [chat_interface.chatbot_state], [fake_response_textbox], queue=False, trigger_mode="always_last")
            fake_response_textbox.change(display_attribution_fn, [fake_response_textbox, doc_text_dict], [attribution_chunks_html, tabs], queue=False, trigger_mode="always_last")

    demo.queue(api_open=False).launch(
        server_name=get_local_ip(), server_port=port, 
        share=False, show_api=False, 
        favicon_path=Path(__file__).parent / "fig/favicon.png", 
        auth=[("xinshuohu", "emnlp2024@demo"), ("emnlp2024@demo", "emnlp2024@demo")], auth_message="""The <b><i>username</i></b> and <b><i>password</i></b> can be found in the footnote of our underview paper."""
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed_model_path", type=str, default=None)
    parser.add_argument("--orc_model_path", type=str, default=None)
    parser.add_argument("--chat_model_path", type=str, default=None)
    parser.add_argument("--chat_model_server", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--tmp_folder", type=str, default=None)
    parser.add_argument("--port", type=int, default=8080)

    args = parser.parse_args()

    main(**vars(args))
