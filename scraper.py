import requests
from bs4 import BeautifulSoup
import os
import pandas as pd
from urllib.parse import urljoin

# Configuration
input_file = "confluence_urls.csv"  
output_dir = "./scraped_data"
os.makedirs(output_dir, exist_ok=True)

# Load URLs
urls_df = pd.read_csv(input_file)

for idx, row in urls_df.iterrows():
    target_url = row['url']
    print(f"Scraping: {target_url}")

    try:
        response = requests.get(target_url, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract text content
        text_content = soup.get_text(separator='\n', strip=True)

        # Extract image URLs
        image_elements = soup.find_all('img')
        image_urls = [img['src'] for img in image_elements if 'src' in img.attrs]

        # Prepare folder per URL
        safe_page_id = target_url.replace("https://", "").replace("/", "_")
        page_dir = os.path.join(output_dir, safe_page_id)
        os.makedirs(page_dir, exist_ok=True)

        # Save text content
        with open(os.path.join(page_dir, "page.txt"), "w", encoding="utf-8") as f:
            f.write(text_content)

        # Save HTML content
        with open(os.path.join(page_dir, "page.html"), "w", encoding="utf-8") as f:
            f.write(response.text)

        # Download images
        image_folder = os.path.join(page_dir, "images")
        os.makedirs(image_folder, exist_ok=True)

        for img_idx, img_url in enumerate(image_urls):
            # Handle relative image URLs
            full_img_url = urljoin(target_url, img_url)

            try:
                img_data = requests.get(full_img_url, timeout=15).content
                img_filename = f"image_{img_idx}.jpg"
                with open(os.path.join(image_folder, img_filename), "wb") as img_file:
                    img_file.write(img_data)

            except Exception as img_err:
                print(f"Failed to download image {full_img_url}: {img_err}")

    except Exception as e:
        print(f"Failed to scrape {target_url}: {e}")
