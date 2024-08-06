import os
import requests
import logging
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

import gradio as gr

from .html_parser import parse_html_text, parse_html_title
from .pdf_parser import PDFProcesser
from .file_loader import load_file


def is_pdf(content: str):
    return bool(content.startswith('%PDF'))


def extract_url_last_level(url: str) -> str:
    path = urlsplit(url).path
    last_level = path.split('/')[-1]
    return last_level


def rename_file(old_file_path: str, new_file_name: str) -> str:
    dir_name = os.path.dirname(old_file_path)
    old_name = os.path.basename(old_file_path)

    new_path = os.path.join(dir_name, new_file_name)
    
    os.rename(old_file_path, new_path)
    return new_path


def get_file_path(url: str):
    temp_dir = os.environ.get("GRADIO_TEMP_DIR") or str(Path(tempfile.gettempdir()) / "gradio")
    temp_dir = os.path.join(temp_dir, "PDF")
    os.makedirs(temp_dir, exist_ok=True)

    tmp_file_path = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir).name
    file_name = extract_url_last_level(url)
    file_name = "{}.pdf".format(file_name) if not file_name.endswith(".pdf") else file_name
    file_path = rename_file(tmp_file_path, file_name)

    return file_path


def parse_web_pdf(content: bytes, url: str, pdf_processer: PDFProcesser):
    file_path = get_file_path(url)

    with open(file_path, "wb") as fp: 
        fp.write(content)
    
    logging.info("Save web PDF file to: {}".format(file_path))
    
    document = load_file(file_path, pdf_processer)
    
    return document


def load_url(url, pdf_processer=None):
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        content = response.text

        document = None
        if is_pdf(content):
            logging.info("Parse web PDF file from: {}".format(url))
            document = parse_web_pdf(response.content, url, pdf_processer)
            title = ""

        if document is None or len(document) == 0:
            document = parse_html_text(content)
            title = parse_html_title(content)
    except Exception as e:
        logging.error(e)
        gr.Warning("Cannot Load Web Page: {}".format(url))
        document = ""
        title = ""
    return document, title

