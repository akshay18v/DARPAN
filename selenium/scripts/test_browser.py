# Save this as test_browser.py in selenium/scripts/
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

print("Starting Browser...")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
driver.get("https://www.airbnb.co.in")
print("Page Title is:", driver.title)
driver.quit()
print("Success!")