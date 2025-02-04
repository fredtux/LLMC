from goose3 import Goose
import os
from pathlib import Path
import logging
import json
from typing import Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('processing.log'),
        logging.StreamHandler()
    ]
)


def extract_article_content(html_path: str) -> Optional[Dict]:
    """
    Extract title and content from HTML file using Goose3
    """
    try:
        g = Goose({'strict': False})
        with open(html_path, 'r', encoding='utf-8') as file:
            html_content = file.read()

        article = g.extract(raw_html=html_content)
        return {
            'title': article.title,
            'content': article.cleaned_text,
            'metadata': {
                'original_file': os.path.basename(html_path)
            }
        }
    except Exception as e:
        logging.error(f"Error processing {html_path}: {str(e)}")
        return None
    finally:
        g.close()


def process_files(base_dir: str):
    """
    Process all HTML files in the directory structure
    """
    # Get all regions (Ardeal, Muntenia)
    regions = [d for d in os.listdir(base_dir)
               if os.path.isdir(os.path.join(base_dir, d))
               and not d.endswith('_cleaned')]

    for region in regions:
        region_path = os.path.join(base_dir, region)
        # Create cleaned region directory
        cleaned_region_path = os.path.join(base_dir, f"{region}_cleaned")
        os.makedirs(cleaned_region_path, exist_ok=True)

        # Process each county (judet)
        for judet in os.listdir(region_path):
            judet_path = os.path.join(region_path, judet)
            if not os.path.isdir(judet_path):
                continue

            # Create cleaned county directory
            cleaned_judet_path = os.path.join(cleaned_region_path, judet)
            os.makedirs(cleaned_judet_path, exist_ok=True)

            logging.info(f"Processing files in {judet_path}")

            # Process each HTML file
            for html_file in os.listdir(judet_path):
                if not html_file.endswith('.html'):
                    continue

                html_path = os.path.join(judet_path, html_file)
                output_base = os.path.splitext(html_file)[0]
                output_path = os.path.join(cleaned_judet_path, f"{output_base}.json")

                logging.info(f"Processing {html_file}")

                # Extract content
                article_data = extract_article_content(html_path)

                if article_data:
                    # Save extracted content as JSON
                    try:
                        with open(output_path, 'w', encoding='utf-8') as f:
                            json.dump(article_data, f, ensure_ascii=False, indent=2)
                        logging.info(f"Successfully saved {output_path}")
                    except Exception as e:
                        logging.error(f"Error saving {output_path}: {str(e)}")
                else:
                    logging.warning(f"No content extracted from {html_file}")


def main():
    """
    Main function to run the processor
    """
    # Get the current working directory
    base_dir = os.getcwd()

    logging.info("Starting HTML processing")
    process_files(base_dir)
    logging.info("Finished processing")


if __name__ == "__main__":
    main()