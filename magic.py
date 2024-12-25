import csv
import os
import random
import json
import logging
from selenium.webdriver.chrome.service import Service
from seleniumwire import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import re
import trafilatura
from openai import Client
from datetime import datetime
import time
import platform
from retrying import retry
from typing import List
from concurrent.futures import ThreadPoolExecutor

# Load settings from settings.json
with open('settings.json', 'r') as f:
    settings = json.load(f)

OPENAI_API_KEY = settings["openai_api_key"]
OUTPUT_DIRECTORY = settings["output_directory"]
KEYWORDS_FILE = settings["keywords_file"]

# Setup basic configuration for logging
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Get the current time to include in the log file name
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Setup basic configuration for logging
log_file = os.path.join(log_dir, f'scraper_{current_time}.log')
logging.basicConfig(level=logging.INFO,
                    filename=log_file,
                    filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    encoding='utf-8')

logging.getLogger('selenium').setLevel(logging.WARNING)
logging.getLogger('seleniumwire').setLevel(logging.WARNING)


def get_chromedriver_path():
    system = platform.system()
    if system == 'Windows':
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chromedriver.exe')
    elif system == 'Darwin':
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chromedriver')
    elif system == 'Linux':
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chromedriver')
    else:
        raise Exception(f"Unsupported OS: {system}")


def setup_webdriver(proxy=None):
    chromedriver_path = get_chromedriver_path()
    service = Service(executable_path=chromedriver_path)
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920x1080")

    selenium_wire_options = None

    try:
        driver = webdriver.Chrome(
            service=service, options=chrome_options, seleniumwire_options=selenium_wire_options)
        logging.info("WebDriver initialized successfully.")
    except Exception as e:
        logging.error("Failed to initialize WebDriver: %s", e)
        raise SystemExit(e)
    return driver


@retry(stop_max_attempt_number=3, wait_fixed=2000)
def fetch_url_with_retry(url):
    downloaded = trafilatura.fetch_url(url)
    return downloaded


def clean_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_with_trafilatura(url):
    logging.info("Extracting content from URL: %s", url)
    try:
        downloaded = fetch_url_with_retry(url)
        if downloaded:
            content = trafilatura.extract(
                downloaded, include_comments=False, include_tables=False)
            if content:
                cleaned_content = clean_text(content)
                if len(cleaned_content) > 400:
                    logging.info(
                        "Content extracted successfully from URL: %s", url)
                    return cleaned_content
                else:
                    logging.warning(
                        "Content from URL: %s is less than 400 characters, skipping.", url)
                    return None
            else:
                logging.warning("Failed to extract content from URL: %s", url)
        else:
            logging.warning("Failed to download content from URL: %s", url)
    except Exception as e:
        logging.error("Error extracting content from URL %s: %s", url, e)
    return None


def get_links_with_beautifulsoup(driver):
    logging.info("Retrieving page source for URL extraction")
    skip_domains = ["reddit.com", "istockphoto.com", "groupon.com", "youtube.com",
                    "tiktok.com", "facebook.com", "twitter.com", "petsmart.com", "linkedin.com"]
    skip_extensions = [".gov", ".in", ".uk"]
    skip_words = ["collections", "login", "signin", "advisor", "gov"]

    try:
        html = driver.page_source
        logging.info("Page source retrieved successfully.")
    except Exception as e:
        logging.error("Failed to retrieve page source: %s", e)
        return []

    try:
        soup = BeautifulSoup(html, 'lxml', from_encoding='utf-8')
        logging.info("HTML parsed successfully with BeautifulSoup.")
    except Exception as e:
        logging.error("Failed to parse HTML with BeautifulSoup: %s", e)
        return []

    links = soup.find_all('a', attrs={'jsname': 'UWckNb'})
    logging.info("Found %d links in the page source.", len(links))
    urls = [link.get('href') for link in links[:8]]  # Get the first 8 URLs
    logging.info("Extracted initial %d URLs.", len(urls))

    filtered_urls = [
        url for url in urls
        if not any(domain in url for domain in skip_domains)
        and not any(url.endswith(ext) for ext in skip_extensions)
        and not any(word in url for word in skip_words)
    ]
    logging.info("Filtered URLs count after initial filter: %d",
                 len(filtered_urls))

    additional_attempts = 0
    while len(filtered_urls) < 5 and len(filtered_urls) < len(links):
        additional_attempts += 1
        more_links = links[len(filtered_urls):len(filtered_urls) + 5]
        more_urls = [link.get('href') for link in more_links]
        filtered_urls += [
            url for url in more_urls
            if not any(domain in url for domain in skip_domains)
            and not any(url.endswith(ext) for ext in skip_extensions)
            and not any(word in url for word in skip_words)
        ]
        logging.info("Filtered URLs count after additional filtering attempt %d: %d",
                     additional_attempts, len(filtered_urls))
        if additional_attempts > 10:
            logging.warning(
                "Reached maximum number of additional filtering attempts.")
            break

    filtered_urls = filtered_urls[:5]  # Limit to 5 URLs
    for url in filtered_urls:
        logging.info("Retained URL: %s", url)

    if not filtered_urls:
        logging.warning("No URLs retained after filtering.")
    else:
        logging.info("Filtered URLs: %s", filtered_urls)

    return filtered_urls


# Initialize OpenAI client
client = Client(api_key=OPENAI_API_KEY)


def summarize_text(text):
    logging.info("Summarizing text with OpenAI GPT-4o-mini ")
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You excel in extracting context and key information from long pieces of web page data provided to you. You ignore any mention of the website, any promotional material. You retain key information guides, crucial information or information that readers love and need to know etc. You remove headings, footers, copyright notices etc. You don't return any other information. Just the key information in a list form."
                },
                {
                    "role": "user",
                    "content": f"""If the text is provided in another language, translate it to English. I will provide you with a text. Your task is to process the information and present it in the following format:

        1. SHORTLY: Summarize the main idea of the text in 1-2 sentences.
        2. MORE: Provide a more detailed explanation of the key points from the text in a few sentences, using clear and concise language.
        3. IMPORTANT: Highlight 5 important facts or statistics from the text, presented in bullet points.
        4. RELATED: Suggest 5 related questions or topics based on the content of the text.

        return this all in a dictionary (example: {{
            "shortly": "SHORTLY",
            "more": "MORE",
            "important": "IMPORTANT",
            "related": "RELATED"
        }})

        Now, process the following text in this format: Extract key information from the following text: {text}"""
                }
            ],
            temperature=0.7,  # Adjust the creativity of the response
            max_tokens=1000,  # Limit the length of the summary
            top_p=1.0,
            frequency_penalty=0.5,
            presence_penalty=0.5
        )

        summary = completion.choices[0].message.content.strip()
        return json.loads(summary)
    except Exception as e:
        logging.error(
            f"Failed to summarize text with OpenAI GPT-4o-mini Turbo: %s", e)
        return None


def process_keyword(keyword):
    driver = setup_webdriver()
    driver.get(f'https://www.google.com/search?q={keyword}')
    links = get_links_with_beautifulsoup(driver)
    contents = extract_contents_from_links(links)
    save_contents_to_json(keyword, contents)
    driver.quit()


def extract_contents_from_links(links: List[str]):
    all_text = []
    for link in links:
        content = extract_with_trafilatura(link)

        if content:
            all_text.append(content)

    if all_text:
        combined_text = "\n\n".join(all_text)
        summary = summarize_text(combined_text)
        if summary:
            return {
                "urls": links,
                "shortly": summary["shortly"],
                "more": summary["more"],
                "important": summary["important"],
                "related": summary["related"]
            }
    return None


def save_contents_to_json(keyword, contents):
    if contents:
        os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

        if len(keyword) > 20:
            keyword = keyword[:20]

        file_path = os.path.join(OUTPUT_DIRECTORY, f'{keyword}.json')
        with open(file_path, 'w', encoding='utf-8') as outfile:
            json.dump(contents, outfile, indent=4, ensure_ascii=False)
        logging.info(f"Data for keyword {keyword} successfully written to {file_path}")


def main():
    logging.info("Script started")
    try:

        if not os.path.exists(KEYWORDS_FILE):
            logging.error("%s not found. Exiting script.", KEYWORDS_FILE)
            return

        with open(KEYWORDS_FILE, 'r', encoding='utf-8') as f:
            keywords = [row[0]
                        for row in csv.reader(f) if row]  # Assumes no header

        for keyword in keywords:
            logging.info("Processing keyword: %s", keyword)
            try:
                process_keyword(keyword)
            except Exception as e:
                logging.error(
                    "An error occurred while processing keyword %s: %s", keyword, e)
            finally:
                # Introduce a delay to reduce the chance of proxy bans
                time.sleep(random.uniform(1, 3))

            # Remove the processed keyword from the list
            keywords.remove(keyword)
            with open(KEYWORDS_FILE, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows([[k] for k in keywords])

    except Exception as e:
        logging.error("An unexpected error occurred: %s", e)
    logging.info("Script completed")

    # print(extract_contents_from_links(links=['https://en.wikipedia.org/wiki/Cat', 'https://ru.wikipedia.org/wiki/%D0%9A%D0%BE%D1%88%D0%BA%D0%B0']))
    # print(extract_with_trafilatura(
    #     url='https://ru.wikipedia.org/wiki/%D0%9A%D0%BE%D1%88%D0%BA%D0%B0'))
    #  print(summarize_text(tetx))


    # print(process_keyword('Брєд Пітт'))
if __name__ == "__main__":
    main()
