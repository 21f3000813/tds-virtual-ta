# scrape_discourse_topics.py
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time, json, os

def scrape_discourse_topics():
    url = "https://discourse.onlinedegree.iitm.ac.in/c/courses/tds-kb/34"
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(url)
    time.sleep(5)

    for _ in range(5):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(3)

    topic_elements = driver.find_elements(By.CSS_SELECTOR, "a.title.raw-link.raw-topic-link")
    topics = [{"title": el.text, "link": el.get_attribute("href")} for el in topic_elements if el.text and el.get_attribute("href")]

    driver.quit()

    os.makedirs("data", exist_ok=True)
    with open("data/discourse_topics.json", "w", encoding="utf-8") as f:
        json.dump(topics, f, indent=2)

    print(f"Saved {len(topics)} topics to data/discourse_topics.json")

if __name__ == "__main__":
    scrape_discourse_topics()
