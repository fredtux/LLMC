from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from time import sleep
import requests
import os

website_name = input("Ziarul: ") # example : dezvaluiri.ro

# Define categories with their URLs
categories = {
    '1': {
        'main': 'https://www.dezvaluiri.ro/category/actualitate/',
        'page': 'https://www.dezvaluiri.ro/category/actualitate/page/'
    },
    '2': {
        'main': 'https://www.dezvaluiri.ro/category/reportaj/',
        'page': 'https://www.dezvaluiri.ro/category/reportaj/page/'
    },
    '3': {
        'main': 'https://www.dezvaluiri.ro/category/administratie-locala/',
        'page': 'https://www.dezvaluiri.ro/category/administratie-locala/page/'
    },
    '4': {
        'main': 'https://www.dezvaluiri.ro/category/anunturi/',
        'page': 'https://www.dezvaluiri.ro/category/anunturi/page/'
    },
    # Add more categories as needed
}

links_file = f'{website_name}_links.txt'
failed_links_file = f'{website_name}_failed_links.txt'

output_directory = 'Dobrogea/Constanta'
headers = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36'}

os.makedirs(output_directory, exist_ok=True)

# Initialize WebDriver once
driver = webdriver.Chrome()

# classic one = //div[contains(@class,'fc-dialog fc-choice-dialog')]//button[contains(@class,'fc-cta-consent')]

# Function to handle cookie acceptance
def accept_cookies(driver):
    try:
        wait = WebDriverWait(driver, 2)
#         accept_button = wait.until(EC.element_to_be_clickable(
#             (By.XPATH, "//a[@class='cc-btn cc-dismiss' and @aria-label='dismiss cookie message']")
# ))
        accept_button = wait.until(EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler")))
        accept_button.click()
        print("Clicked the 'Accept Cookies' button.")
    except Exception as e:
        print(f"Could not click the 'Accept Cookies' button: {e}")


# Function to get links from a page based on the specified format
def get_links_from_page(url, headers):
    links = []
    try:
        page = requests.get(url, headers=headers)
        soup = BeautifulSoup(page.content, 'html.parser')
        article_links = soup.find_all('li', class_='infinite-post')
        for link in article_links:
            a_tag = link.find('a')
            if a_tag and a_tag.get('href'):
                links.append(a_tag['href'])
            # if link.get('href'):  # Ensure the link has an href attribute
            #     # Append the full URL by combining the base URL and the relative path
            #     full_url = requests.compat.urljoin(url, link['href'])
            #     links.append(full_url)
    except Exception as e:
        print(f"Error getting links from page: {e}")
    return links


# Collect all links from all categories
def get_all_links(categories, headers):
    all_links = []
    cookies_accepted = False

    for category_name, urls in categories.items():
        print(f"\nProcessing category: {category_name}")

        # Handle main page with Selenium (for cookie acceptance)
        driver.get(urls['main'])

        # Accept cookies only once
        if not cookies_accepted:
            accept_cookies(driver)
            cookies_accepted = True
            sleep(2)

        # Get links from main page
        main_page_links = get_links_from_page(urls['main'], headers)
        print(f"Got {len(main_page_links)} links from main page of {category_name}")
        all_links.extend(main_page_links)

        # Get links from additional pages
        for i in range(1, 6):  # Adjust range as needed
            print(f"Getting links from page {i} of {category_name}...")
            sleep(2)
            page_links = get_links_from_page(urls['page'] + str(i), headers)
            print(f"Added {len(page_links)} links from page {i}")
            all_links.extend(page_links)

    # Remove duplicates while preserving order
    all_links = list(dict.fromkeys(all_links))
    return all_links

# Collect all links
all_links = get_all_links(categories, headers)

# Save all links to file
with open(links_file, 'w', encoding='utf-8') as file:
    for link in all_links:
        file.write(link + '\n')

print(f"\nTotal unique links across all categories: {len(all_links)}")

# Initialize counters for downloading
link_no = 0
link_exception = 0
failed_links = []

# Download articles
for link in all_links:
    link_no += 1
    if link_no % 5 == 0:
        print(f"Getting article {link_no}/{len(all_links)} (errors so far {link_exception})")
        sleep(1)

    try:
        response = requests.get(link, headers=headers)
        filename = f"{website_name}_{link_no:04d}.html"
        filepath = os.path.join(output_directory, filename)

        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(response.text)

    except Exception as e:
        print(f"Error fetching {link}: {e}")
        link_exception += 1
        failed_links.append(link)

print(f"Successfully retrieved {len(all_links) - len(failed_links)} articles.")

# Save failed links
with open(failed_links_file, 'w', encoding='utf-8') as file:
    for link in failed_links:
        file.write(link + '\n')

driver.quit()