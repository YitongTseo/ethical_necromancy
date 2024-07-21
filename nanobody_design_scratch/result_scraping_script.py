# conda activate ethical_necromancy
# Check here for the URLs: https://wenmr.science.uu.nl/haddock2.4/workspace

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.action_chains import ActionChains
import requests
import re
import os
import time

import pandas as pd
import pdb

ROOT = "/Users/yitongtseo/Documents/GitHub/ethical_necromancy/nanobody_design_scratch/pdb_files/HADDOCK_results_myosin_head"
# Setup DataFrame
columns = [
    "Target PDB",
    "URL",
    "Cluster ID",
    "Cluster Ranking",
    "HADDOCK score",
    "Cluster size",
    "RMSD from the overall lowest-energy structure",
    "Van der Waals energy",
    "Electrostatic energy",
    "Desolvation energy",
    "Restraints violation energy",
    "Buried Surface Area",
    "Z-Score",
    "HADDOCK Result PDB Filename",
]
results_df = pd.DataFrame(columns=columns)
TOP_N_CLUSTERS = 5


chromedriver_path = "/Users/yitongtseo/Documents/GitHub/ethical_necromancy/nanobody_design_scratch/chromedriver-mac-x64/chromedriver"
service = Service(executable_path=chromedriver_path)
driver = webdriver.Chrome(service=service)


urls_df = pd.read_csv("HADDOCK_progress_urls_myosinheads.csv")
df = pd.DataFrame()
# # URL of the page to scrape (you might need to adjust this)
# url = 'http://example.com/page'
wait = WebDriverWait(driver, 10)

for idx, row in urls_df.iterrows():
    url = row["URL"].replace("run", "result")
    print(url)
    # Fetch the page
    driver.get(url)

    # Wait for JavaScript to load
    WebDriverWait(driver, 20).until(
        EC.visibility_of_element_located((By.CLASS_NAME, "textblock"))
    )

    # Now you can scrape the content
    clusters = driver.find_elements(By.CLASS_NAME, "textblock")

    for cluster_ranking, cluster in enumerate(clusters):
        if cluster_ranking > TOP_N_CLUSTERS:
            continue
        title = cluster.find_element(By.TAG_NAME, "p").text
        cluster_number = re.search(r"Cluster (\d+)", title)
        if cluster_number:
            cluster_number = cluster_number.group(1)
        else:
            cluster_number = "Unknown"

        cluster_data = {
            "Target PDB": url.split("-")[-1],
            "URL": driver.current_url,
            "Cluster ID": cluster_number,
            "Cluster Ranking": cluster_ranking,
        }
        print("Cluster Data:")
        # Extract cluster data
        table = cluster.find_element(By.TAG_NAME, "table")
        rows = table.find_elements(By.TAG_NAME, "tr")
        for row in rows:
            cells = row.find_elements(By.TAG_NAME, "td")
            if len(cells) == 2:
                key = cells[0].text.strip()
                value = cells[1].text.strip()
                cluster_data[key] = value
                print(f"{key}: {value}")

        driver.execute_script("arguments[0].scrollIntoView(true);", cluster)
        time.sleep(1)  # Allow some time for the browser to scroll

        dropdown = cluster.find_element(By.CLASS_NAME, "dropdown-toggle")
        ActionChains(driver).move_to_element(dropdown).click(dropdown).perform()

        # Wait for the dropdown to open and links to be visible
        try:
            wait.until(EC.visibility_of_element_located((By.LINK_TEXT, "PDB Format")))
            pdb_link = cluster.find_element(By.LINK_TEXT, "PDB Format")
            pdb_url = pdb_link.get_attribute("href")

            # Download the file
            pdb_response = requests.get(pdb_url)
            result_pdb_filename = os.path.join(
                ROOT, f"{cluster_data['Target PDB']}_cluster{cluster_ranking}.pdb"
            )
            with open(result_pdb_filename, "wb") as f:
                f.write(pdb_response.content)
            cluster_data["HADDOCK Result PDB Filename"] = result_pdb_filename

        except Exception as e:
            cluster_data["HADDOCK Result PDB Filename"] = "Failed to download"
            print(f"Error downloading: {e}")

        results_df = pd.concat(
            [results_df, pd.DataFrame([cluster_data])], ignore_index=True
        )
        print("------\n")

    results_df.to_csv("myosin_head_haddock_results_myosin_heads.csv", index=False)

# Close the browser
driver.quit()
