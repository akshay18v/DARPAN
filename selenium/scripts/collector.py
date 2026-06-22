import os
import time
from selenium import webdriver

# 1. Targets
SITES = {
    "1": {"name": "airbnb", "url": "https://www.airbnb.co.in/"},
    "2": {"name": "makemytrip", "url": "https://www.makemytrip.com/"},
    "3": {"name": "goibibo", "url": "https://www.goibibo.com/"},
    "4": {"name": "yatra", "url": "https://www.yatra.com/"},
    "5": {"name": "cleartrip", "url": "https://www.cleartrip.com/"}
}

# 2. Selection
print("Which website are you capturing today?")
for key, info in SITES.items():
    print(f"{key}. {info['name'].capitalize()}")

choice = input("Enter number (1-5): ")
if choice not in SITES:
    exit()

site_name = SITES[choice]["name"]
target_url = SITES[choice]["url"]

# 3. Dynamic Folder
BASE_DIR = f"../../data/raw/{site_name}"
os.makedirs(BASE_DIR, exist_ok=True)

# --- STEP 4: SMART COUNT (REPLACES count = 0) ---
# Get list of all png files in the folder that start with the site name
existing_files = [f for f in os.listdir(BASE_DIR) if f.startswith(site_name) and f.endswith('.png')]

if not existing_files:
    count = 0
else:
    # 1. f.split('_')[-1] gets '0.png'
    # 2. .split('.')[0] gets '0'
    # 3. int() converts it to a number
    indices = [int(f.split('_')[-1].split('.')[0]) for f in existing_files]
    count = max(indices) + 1

print(f"\n🚀 Target locked: {site_name.upper()}")
print(f"📂 Found {len(existing_files)} files. Starting from index: {count}")
# -----------------------------------------------

# 5. Start Browser
driver = webdriver.Chrome()
driver.maximize_window()
driver.get(target_url)

print("Press ENTER to capture, 'q' to quit.")

# 6. Capture Loop
while True:
    user_input = input()

    if user_input.lower() == 'q':
        break

    try:
        driver.switch_to.window(driver.window_handles[-1])
        time.sleep(1) 

        filename = f"{site_name}_{count}.png"
        path = os.path.join(BASE_DIR, filename)

        driver.save_screenshot(path)
        print(f"📸 Saved: {path}")

        count += 1
    except Exception as e:
        print(f"Error: {e}")
        break

driver.quit()
print("Session ended. Great job, bhai!")