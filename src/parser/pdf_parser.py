import io
from typing import Optional, List
from pathlib import Path

import torch
import fitz
from PIL import Image
from torch.cuda.amp import autocast
from transformers import NougatProcessor, VisionEncoderDecoderModel
from tqdm import tqdm
from optimum.onnxruntime import ORTModelForVision2Seq
# ! pip install pymupdf, frontend, optimum[onnxruntime[gpu]]


class PDFProcesser:
    def __init__(self, model_name_or_path: str):
        self.processor = NougatProcessor.from_pretrained(model_name_or_path, device_map="auto")
        self.model = ORTModelForVision2Seq.from_pretrained(
            model_name_or_path,
            provider="CUDAExecutionProvider", # 'CUDAExecutionProvider' for gpu 
            use_merged=False,
            use_io_binding=True
        )
        # self.model = VisionEncoderDecoderModel.from_pretrained(model_name_or_path)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def convert_pdf2markdown(self, file_path, max_pages=100):
        images = self._rasterize_paper(pdf=file_path, return_pil=True)

        pages = []
        for image in tqdm(images[:max_pages]):
            with autocast():
                image = Image.open(image)
                pixel_values = self.processor(images=image, return_tensors="pt").pixel_values

                outputs = self.model.generate(
                    pixel_values.to(self.model.device),
                    min_length=1,
                    num_beams=1,
                    do_sample=False,
                    max_new_tokens=3584,
                    bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                )
            
            page = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            page = self.processor.post_process_generation(page, fix_markdown=True)
            pages.append(page.replace(r"\(", "$").replace(r"\)", "$").replace(r"\[", "$$").replace(r"\]", "$$"))
        return " ".join(pages)
    
    def _rasterize_paper(
            self,
            pdf: Path,
            outpath: Optional[Path] = None,
            dpi: int = 96,
            return_pil=False,
            pages=None,
        ) -> Optional[List[io.BytesIO]]:
        """
        Rasterize a PDF file to PNG images.

        Args:
            pdf (Path): The path to the PDF file.
            outpath (Optional[Path], optional): The output directory. If None, the PIL images will be returned instead. Defaults to None.
            dpi (int, optional): The output DPI. Defaults to 96.
            return_pil (bool, optional): Whether to return the PIL images instead of writing them to disk. Defaults to False.
            pages (Optional[List[int]], optional): The pages to rasterize. If None, all pages will be rasterized. Defaults to None.

        Returns:
            Optional[List[io.BytesIO]]: The PIL images if `return_pil` is True, otherwise None.
        """

        pillow_images = []
        if outpath is None:
            return_pil = True
        try:
            if isinstance(pdf, (str, Path)):
                pdf = fitz.open(pdf)
            if pages is None:
                pages = range(len(pdf))
            for i in pages:
                page_bytes: bytes = pdf[i].get_pixmap(dpi=dpi).pil_tobytes(format="PNG")
                if return_pil:
                    pillow_images.append(io.BytesIO(page_bytes))
                else:
                    with (outpath / ("%02d.png" % (i + 1))).open("wb") as f:
                        f.write(page_bytes)
        except Exception as e:
            print(e)
        if return_pil:
            return pillow_images
