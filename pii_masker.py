
from PyPDF2 import PdfReader
import docx
from docx2pdf import convert

import fitz # Import fitz here
import re
import zipfile
import os
import tempfile
from random_generator import generate_random_mobile, generate_random_email, generate_random_address
from urllib.parse import unquote
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PII_MASKER")

def converter(input_docx, output_pdf):
    convert(input_docx, output_pdf)

def extract_mobile_numbers(resume_text):
    pattern = re.compile(
        r'(?:\(?\+91\)?[\s\-]?|91[\s\-]?|0)?[6-9]\d{2}[\s\-]?\d{3}[\s\-]?\d{4}'  # Indian standard format
        r'|[6-9]\d{4}[\s\-]?\d{5}'                                              # Indian split format (e.g., 74181 46722)
        r'|(?:\(?\+1\)?[\s\-]?|1[\s\-]?)?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}'    # US format
    )
    matches = pattern.finditer(resume_text)
    cleaned = [m.group().strip() for m in matches if m.group().strip()]
    return cleaned if cleaned else None

def extract_email_addresses(resume_text):
    pattern = re.compile(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
        )
    matches = pattern.findall(resume_text)
    emails = [e.lower() for e in matches]
    return emails if emails else None

def mask_profile_photo(page, gender):
    image_infos = page.get_image_info()
    if gender.lower() == 'male':
        avatar_path = '/content/drive/MyDrive/CDS-B9-Group11/Capstone Project/male_bitemoji.png'
    else:
        avatar_path = '/content/drive/MyDrive/CDS-B9-Group11/Capstone Project/female_bitemoji.png'
    for info in image_infos:
        x0, y0, x1, y1 = info["bbox"]
        width = x1 - x0
        height = y1 - y0
        page.draw_rect(fitz.Rect(x0, y0, x1, y1), color=(1, 1, 1), fill=(1, 1, 1))

        # Step 2: Insert avatar image
        page.insert_image(
            fitz.Rect(x0, y0, x1, y1),
            filename=avatar_path,
            keep_proportion=True,
            overlay=True
        )
        logger.info(f"Replaced image at ({x0}, {y0}, ({x1}, {y1})) with avatar")
        return True

    return False

def extract_hyperlinks(text, annotations=None):
    links = []
    # 1. Match visible URLs
    url_pattern = r'\b(?:https?://|www\.)[^\s<>()]+'
    links.extend(re.findall(url_pattern, text))

    # 2. Match mailto: links
    mailto_pattern = r'mailto:[^\s<>()]+'
    links.extend(re.findall(mailto_pattern, text))

    # 3. Match file:/// URIs
    file_uri_pattern = r'file:///[^\s<>()]+'
    links.extend(re.findall(file_uri_pattern, text))

    # 4. Extract embedded hyperlinks from PDF annotations (if provided)
    if annotations:
        for annot in annotations:
            uri = getattr(annot, "uri", None)
            if uri:
                decoded = unquote(uri)
                links.append(decoded)

    # 5. Normalize and deduplicate
    clean_links = set()
    for link in links:
        # Strip trailing punctuation
        link = link.rstrip('.,);]')
        clean_links.add(link)
    return list(clean_links)

def classify_link(url):
    if url.startswith("file:///"):
        if "linkedin.com" in url:
            return "LinkedIn"
        elif "github.com" in url:
            return "GitHub"
        else:
            return "LocalFile"
    elif "github.com" in url:
        return "GitHub"
    elif "linkedin.com" in url:
        return "LinkedIn"
    elif "stackoverflow.com" in url:
        return "StackOverflow"
    elif "mailto:" in url or "@" in url:
        return "Email"
    else:
        return "Other"

def generate_dummy_link(link_type):
    dummy_links = {
        "GitHub": "https://github.com/username",
        "LinkedIn": "https://linkedin.com/in/profile",
        "StackOverflow": "https://stackoverflow.com/users/123456",
        "Email": "mailto:someone@example.com",
        "LocalFile": "https://example.com/localfile",
        "Other": "https://example.com"
    }
    return dummy_links.get(link_type, "https://example.com")


def extract_address(resume_text):
    pattern = re.compile(
        r'('
        # 🏙️ Street-style addresses
        r'\b\d{1,5}\s[\w\s]{1,40}'
        r'(Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Boulevard|Blvd|Drive|Dr|Block|Sector|Colony|Nagar|Enclave)\b.*'
        r')|('
        # 🏢 City-Pincode-State format
        r'\b[A-Z][a-z]+[-\s]?\d{6},?\s?[A-Z][a-z]+\b'
        r')|('
        r'^([\w\s]+),\s*([\w\s]+),\s*([\w\s]+)\s*\((\d{6})\)$'
        r')|('
        # 🏠 Indian-style apartment/locality format
        r'(?:Flat\sNo\.?|#|Apartment\sNo\.?)\s?\d{1,5},?\s*'
        r'(?:[A-Za-z0-9\s\-]{2,40},?\s*)?'
        r'(?:\d{1,2}(?:st|nd|rd|th)?\s(?:Main|Cross|Street|Road),?\s*)?'
        r'(?:[A-Za-z\s]{2,40},?\s*)?'
        r'(?:Bangalore|Hyderabad|Mumbai|Chennai|[A-Za-z\s]{2,30}),?\s*'
        r'(?:Karnataka|Telangana|Maharashtra|Tamil Nadu|[A-Za-z\s]{2,30}),?\s*'
        r'\d{6}'
        r')',
        re.IGNORECASE
    )

    matches = pattern.findall(resume_text)
    return matches

def redact_personal_information(input_pdf, output_pdf=None, save=False):
    logger.info("Starting masking of personal information")
    if isinstance(input_pdf, str):
        gender = input_pdf.split('_')[-2]
        doc = fitz.open(input_pdf)
    else:
        file_name = input_pdf.name
        gender = file_name.split('_')[2]
        doc = fitz.open(stream=input_pdf.read(), filetype="pdf")
    for page in doc:
        mask_profile_photo(page, gender)
        text_dict = page.get_text("dict")

        for block in text_dict["blocks"]:
            if "lines" in block:
                for line in block["lines"]:
                    # Redact other sensitive info span-wise
                    for span in line["spans"]:
                        span_text = span["text"]

                        if extract_email_addresses(span_text):
                            logger.info("Masking Email address")
                            page.add_redact_annot(
                                span["bbox"],
                                text=generate_random_email(),
                                fill=(1.0, 1.0, 1.0)
                            )

                        if extract_mobile_numbers(span_text):
                            logger.info("Masking Mobile number")
                            page.add_redact_annot(
                                span["bbox"],
                                text=generate_random_mobile(),
                                fill=(1.0, 1.0, 1.0)
                            )

                        if extract_address(span_text):
                            logger.info("Masking Address")
                            page.add_redact_annot(
                                span["bbox"],
                                text=generate_random_address(),
                                fill=(1.0, 1.0, 1.0)
                            )

                        for link in extract_hyperlinks(span_text, annotations=page.annots()):
                            logger.info("Masking important hyperlinks")
                            link_type = classify_link(link)
                            dummy = generate_dummy_link(link_type)
                            page.add_redact_annot(span["bbox"], text=dummy, fill=(1.0, 1.0, 1.0))

        page.apply_redactions()

    if save:
        print("save")
        doc.save(output_pdf)
        doc.close()
    else:
        print("Enter")
        from io import BytesIO
        output_stream = BytesIO()
        # After applying redactions
        doc.save(output_stream)
        doc.close()
        # Move to beginning of stream
        output_stream.seek(0)
        # Now you can return, send, or preview this stream
        return output_stream
