import os
import re
import logging

import gradio as gr
import torch
from langchain_community.document_loaders import TextLoader, Docx2txtLoader, PDFMinerLoader, UnstructuredMarkdownLoader

from .pdf_parser import PDFProcesser


def replace_newlines(text: str) -> str:
    text = re.sub(r'\n{2,}', '###NEWLINE###', text)
    # text = re.sub(r'-\n', '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'###NEWLINE###', '\n\n', text)
    return text


def load_file(file_path: str, pdf_processer: PDFProcesser = None) -> str:
    file_name = os.path.basename(file_path)
    file_basic_name, file_extension = os.path.splitext(file_name)

    try:
        if file_extension == ".pdf":
            document = None
            if pdf_processer is not None:
                try:
                    document = pdf_processer.convert_pdf2markdown(file_path)
                except Exception as e:
                    logging.error(e)
                    document = None
                finally:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                            
            if document is None:
                loader = PDFMinerLoader(file_path)
                document = loader.load()[0].page_content
                document = replace_newlines(document)
        elif file_extension == ".docx":
            loader = Docx2txtLoader(file_path)
            document = loader.load()[0].page_content
        elif file_extension == ".md":
            loader = UnstructuredMarkdownLoader(file_path)
            document = loader.load()[0].page_content
        else:
            loader = TextLoader(file_path, encoding="utf-8")
            document = loader.load()[0].page_content
    except Exception as e:
        logging.error(e)
        gr.Warning("Cannot Load File: {}".format(file_name))
        document = ""

    document = re.sub(r'\n{2,}', '\n\n', document)

    return document
