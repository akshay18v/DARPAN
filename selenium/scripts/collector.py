import os
import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# ======================================================
# 1. TARGET WEBSITES
# ======================================================

SITES = {
    "1": {"name": "airbnb", "url": "https://www.airbnb.co.in/"},
    "2": {"name": "makemytrip", "url": "https://www.makemytrip.com/"},
    "3": {"name": "goibibo", "url": "https://www.goibibo.com/"},
    "4": {"name": "yatra", "url": "https://www.yatra.com/"},
    "5": {"name": "cleartrip", "url": "https://www.cleartrip.com/"}
}

# ======================================================
# 2. WEBSITE SELECTION
# ======================================================

print("Which website are you capturing today?")

for key, info in SITES.items():
    print(f"{key}. {info['name'].capitalize()}")

choice = input("Enter number (1-5): ").strip()

if choice not in SITES:
    print("Invalid choice.")
    exit()

site_name = SITES[choice]["name"]
target_url = SITES[choice]["url"]

# ======================================================
# 3. PROJECT PATH (FIXED)
# ======================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

BASE_DIR = PROJECT_ROOT / "data" / "raw" / site_name
BASE_DIR.mkdir(parents=True, exist_ok=True)

# ======================================================
# 4. SMART COUNT
# ======================================================

existing_files = [
    f.name
    for f in BASE_DIR.glob("*.png")
    if f.name.startswith(site_name)
]

if not existing_files:
    count = 0
else:
    indices = []

    for file in existing_files:
        try:
            index = int(file.split("_")[-1].split(".")[0])
            indices.append(index)
        except ValueError:
            pass

    count = max(indices) + 1 if indices else 0

print(f"\n🚀 Target locked: {site_name.upper()}")
print(f"📂 Saving screenshots inside:")
print(BASE_DIR)
print(f"\n📂 Found {len(existing_files)} files.")
print(f"📸 Starting from index: {count}")

# ======================================================
# 5. START CHROME
# ======================================================

print("\nOpening Chrome...")

driver = webdriver.Chrome(
    service=Service(ChromeDriverManager().install())
)

driver.maximize_window()

driver.get(target_url)

print("Chrome opened successfully.")
print("\nPress ENTER to capture.")
print("Type 'q' then ENTER to quit.\n")

# ======================================================
# 6. CAPTURE LOOP
# ======================================================

while True:

    user_input = input()

    if user_input.lower() == "q":
        break

    try:

        driver.switch_to.window(driver.window_handles[-1])

        time.sleep(1)

        filename = f"{site_name}_{count}.png"

        path = BASE_DIR / filename

        success = driver.save_screenshot(str(path))

        if success:
            print(f"📸 Saved: {filename}")
            print(f"📂 Location: {path.resolve()}")
            count += 1
        else:
            print("Screenshot could not be saved.")

    except Exception as e:
        print(f"Error: {e}")
        break

driver.quit()

print("\nSession ended. Great job, bhai!")

