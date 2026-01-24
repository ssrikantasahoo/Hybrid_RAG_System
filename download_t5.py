
import logging
import time
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_model():
    model_name = "google/flan-t5-base"
    logger.info(f"Starting robust download for {model_name}...")
    
    # Ensure hf_hub is using the right transfer method if possible
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    
    max_retries = 5
    for i in range(max_retries):
        try:
            logger.info(f"Attempt {i+1}/{max_retries}")
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Load model
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            
            logger.info("Download completed successfully!")
            return True
        except Exception as e:
            logger.error(f"Download failed on attempt {i+1}: {e}")
            if i < max_retries - 1:
                logger.info("Retrying in 5 seconds...")
                time.sleep(5)
            else:
                logger.error("All retries failed.")
                return False

if __name__ == "__main__":
    download_model()
