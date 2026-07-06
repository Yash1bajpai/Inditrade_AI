"""
DGFT Policy Notification PDF Extractor for IndiTrade AI.

Downloads export-import policy notifications and trade circulars from the
Directorate General of Foreign Trade (DGFT) website.
Extracts clean plain text using pdfplumber for RAG vector ingestion and Q&A dataset generation.
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import requests

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("DGFTExtractor")

# Sample DGFT Notification PDFs (Public URLs or Seed Policy Text if URLs expire/block)
SAMPLE_DGFT_NOTIFICATIONS = [
    {
        "id": "DGFT_Notif_12_2015_20",
        "title": "Amendment in Export Policy of Wheat (HS Code 1001)",
        "date": "2022-05-13",
        "category": "Export Restriction",
        "text_content": """
        TO BE PUBLISHED IN THE GAZETTE OF INDIA EXTRAORDINARY
        GOVERNMENT OF INDIA, MINISTRY OF COMMERCE AND INDUSTRY
        DEPARTMENT OF COMMERCE, DIRECTORATE GENERAL OF FOREIGN TRADE
        
        NOTIFICATION No. 06/2015-2020, New Delhi, Dated: 13th May, 2022
        
        Subject: Amendment in Export Policy of Wheat under HS Code 1001 of Chapter 10 of ITC (HS), 2022.
        
        S.O. (E): In exercise of powers conferred by Section 3 read with Section 5 of the Foreign Trade (Development & Regulation) Act, 1992, as amended, read with Para 1.02 and 2.01 of the Foreign Trade Policy, 2015-20, the Central Government hereby makes the following amendment in the Export Policy of Chapter 10 of ITC (HS) Export Policy:
        
        1. There is a sudden spike in the global prices of wheat arising out of many factors, as a result of which the food security of India, neighbouring and other vulnerable countries is at risk.
        2. The Government of India is committed to providing for the food security requirements of India, neighbouring and other vulnerable developing countries which are adversely affected by the sudden changes in the global market for wheat and are unable to access adequate wheat supplies.
        3. In order to manage the overall food security of the country and to support the needs of neighbouring and other vulnerable countries, the Export Policy of Wheat (HS Code 1001) is revised from 'Free' to 'Prohibited' with immediate effect.
        
        Effect of this Notification: The export of wheat under HS Code 1001 is prohibited with immediate effect. However, export will be allowed on the basis of permission granted by the Government of India to other countries to meet their food security needs and based on the request of their governments.
        """
    },
    {
        "id": "DGFT_Notif_23_2023",
        "title": "Import Policy of Laptops, Tablets, and Personal Computers",
        "date": "2023-08-03",
        "category": "Import Restriction",
        "text_content": """
        GOVERNMENT OF INDIA, MINISTRY OF COMMERCE AND INDUSTRY
        DIRECTORATE GENERAL OF FOREIGN TRADE
        
        NOTIFICATION No. 23/2023, New Delhi, Dated: 3rd August, 2023
        
        Subject: Amendment in Import Policy of Laptops, Tablets, All-in-One Personal Computers, and Ultra small form factor Computers and Servers under HSN 8471.
        
        In exercise of powers conferred by Section 3 and Section 5 of the FT (D&R) Act, 1992, the Central Government hereby makes the following amendments in Chapter 84 of Schedule-I (Import Policy) of ITC (HS) 2022:
        
        1. Import of Laptops, Tablets, All-in-One Personal Computers, and Ultra small form factor Computers and Servers falling under HSN 8471 shall be 'Restricted' and their import would be allowed against a valid Licence for Restricted Imports.
        2. Exemption from Import Licensing requirements is provided for Import of 1 Laptop, Tablet, All-in-One Personal Computer, or Ultra small form factor Computer, including those purchased from e-commerce portals through post or courier. Imports shall be subject to payment of applicable duty.
        3. Exemption from licensing is also provided for up to 20 items per consignment for the purpose of R&D, Testing, Benchmarking, and Evaluation, Repair and return, and Product Development.
        
        Effect of the Notification: Import of Laptops, Tablets, All-in-One PCs, and Servers under HSN 8471 is restricted with immediate effect. Exemption is provided for up to 20 units for R&D and testing purposes.
        """
    },
    {
        "id": "DGFT_Circular_RoDTEP_2024",
        "title": "Extension of Remission of Duties and Taxes on Exported Products (RoDTEP) Scheme",
        "date": "2024-03-08",
        "category": "Export Incentive",
        "text_content": """
        GOVERNMENT OF INDIA, MINISTRY OF COMMERCE AND INDUSTRY
        DIRECTORATE GENERAL OF FOREIGN TRADE
        
        TRADE NOTICE No. 35/2023-24, New Delhi, Dated: 8th March, 2024
        
        Subject: Extension of RoDTEP scheme support for export items and revision of rates.
        
        1. Attention of the trade and industry is invited to the Remission of Duties and Taxes on Exported Products (RoDTEP) Scheme notified vide Notification No. 19/2015-20.
        2. In line with the government's commitment to boost Indian exports and enhance global competitiveness of Indian exporters, it has been decided to extend the support under the RoDTEP scheme to Advance Authorization (AA) holders, Export Oriented Units (EOUs), and Special Economic Zones (SEZs) export units.
        3. The revised budgetary allocation for the RoDTEP scheme ensures that labor-intensive sectors such as textiles, engineering goods, agriculture, and marine products continue to receive robust remission rates to offset non-creditable central, state, and local taxes.
        
        Effect of this Circular: RoDTEP scheme benefits are officially extended to SEZ, EOU, and Advance Authorization export units to support global export growth.
        """
    }
]


class DGFTExtractor:
    """Extracts policy notification text for RAG chunking and Q&A dataset generation."""

    def __init__(self, raw_dir: str = "data/raw/dgft", processed_dir: str = "data/processed"):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extracts plain text from a local PDF file using pdfplumber."""
        if not HAS_PDFPLUMBER:
            logger.warning("pdfplumber not installed. Cannot parse binary PDF.")
            return ""
            
        try:
            text_lines = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        text_lines.append(text)
            return "\n".join(text_lines)
        except Exception as e:
            logger.error(f"Error extracting PDF {pdf_path}: {e}")
            return ""

    def clean_policy_text(self, text: str) -> str:
        """Removes excessive whitespace, headers, and formatting artifacts."""
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"(Page \d+ of \d+|GOVERNMENT OF INDIA)", "", text, flags=re.IGNORECASE)
        return text.strip()

    def run_pipeline(self) -> str:
        """
        Processes available PDFs in raw_dir or uses structured seed notifications.
        Outputs clean text chunks to JSONL for Qdrant RAG ingestion and Q&A generator.
        """
        logger.info("Starting DGFT Policy Notification extraction pipeline...")
        
        extracted_docs = []
        
        # 1. Check local raw directory for downloaded PDFs
        pdf_files = list(self.raw_dir.glob("*.pdf"))
        if pdf_files:
            logger.info(f"Found {len(pdf_files)} PDF files in {self.raw_dir}. Extracting...")
            for pdf_file in pdf_files:
                raw_text = self.extract_text_from_pdf(pdf_file)
                if raw_text:
                    clean_txt = self.clean_policy_text(raw_text)
                    extracted_docs.append({
                        "id": pdf_file.stem,
                        "title": pdf_file.stem.replace("_", " "),
                        "source": "DGFT Notification PDF",
                        "content": clean_txt
                    })
                    
        # 2. If no local PDFs extracted, load seed policy notifications as loud fallback
        if not extracted_docs:
            logger.warning("WARNING: USING SYNTHETIC DATA — REAL SOURCE FAILED (No local PDF files found or extraction failed)")
            print("\n[WARNING: USING SYNTHETIC DATA — REAL SOURCE FAILED]\n")
            logger.info("Loading seed DGFT policy notifications for RAG & Q&A corpus...")
            for notif in SAMPLE_DGFT_NOTIFICATIONS:
                extracted_docs.append({
                    "id": notif["id"],
                    "title": notif["title"],
                    "date": notif["date"],
                    "category": notif["category"],
                    "source": "DGFT Official Gazette (Seed Fallback)",
                    "content": self.clean_policy_text(notif["text_content"])
                })
                
        # Save to JSONL
        output_file = self.processed_dir / "dgft_policy_chunks.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for doc in extracted_docs:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                
        logger.info(f"SUCCESS: Saved {len(extracted_docs)} DGFT policy documents to {output_file}")
        return str(output_file)


if __name__ == "__main__":
    extractor = DGFTExtractor()
    out = extractor.run_pipeline()
    print(f"\n[+] DGFT extraction completed: {out}")
