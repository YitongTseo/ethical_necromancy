# pip install selenium
# https://wenmr.science.uu.nl/haddock2.4/submit/1

# Chrome version 123.0.6312.87
# chrome://settings/help

# consulting this JSON file to try and find a suitable chromedriver
# https://googlechromelabs.github.io/chrome-for-testing/123.0.6312.0.json

# which says to download this chromedriver:
# https://storage.googleapis.com/chrome-for-testing-public/123.0.6312.0/mac-x64/chromedriver-mac-x64.zip

# Here's where we cancel jobs...
# https://wenmr.science.uu.nl/haddock2.4/workspace

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
import csv

import time
import pdb
import getpass

# from 3:21 -->

directory = "/Users/yitongtseo/Documents/GitHub/ethical_necromancy/nanobody_llm_refinement/idx206_8efd"
NANOBODY_CDH_RESIDUES = "26,27,28,29,30,31,32,33,51,52,53,54,55,56,57,58,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120"
# IDX0_MYOSIN_HOTSPOTS = "859,860,861,862,863,864,865,866,949,950,951,1145,1146,1147,1148,1149,1150,1151,1191,1192,1193,1194,1195,1196,1197,1198,1199,1200,1201,1202,1203,1204,1205,1206,1207,1357,1358,1359,1360,1361,1362,1457,1458"
# IDX60_MYOSIN_HOTSPOTS = "1166,1167,1168,1196,1197,1198,1199,1200,1201,1202,1203,1204,1205,1206,1207,1208,1209,1210,1304,1305,1306,1307,1308,1309,1310,1311,1312,1313,1314,1315,1316,1317,1318,1319,1320,1321,1322,1323"
# IDX120_MYOSIN_HOTSPOTS = "804,805,806,807,808,823,824,825,826,827,828,829,830,831,832,833,834,835,836,876,877,878,879,880,881,882,883,884,885,886,887,888,889,890,891,892,893,894,1331,1332,1333,1334,1335,1336,1337,1338"
IDX206_MYOSIN_HOTSPOTS = "1151,1152,1153,1154,1155,1156,1157,1158,1159,1160,1192,1193,1194,1195,1196,1197,1125,1126,1127,1128,1292,1293,1294,1295,1296,1297,1298,1299,1300,1301,1302,1303,1304,1305,1306,1307,1308,1309,1310"

URLS_TO_CHECK_UP_ON = []


user_email = input("What's your email yitong: ")
user_password = getpass.getpass("What's your password: ")

chromedriver_path = "/Users/yitongtseo/Documents/GitHub/ethical_necromancy/nanobody_design_scratch/chromedriver-mac-x64/chromedriver"
service = Service(executable_path=chromedriver_path)
driver = webdriver.Chrome(service=service)
first_run = True
FINISHED_RUNS = [
    "MyoHead_120_E117S_T98N",
    "MyoHead_120_L107R_None",
    "MyoHead_120_S101Q_N104Q",
    "MyoHead_120_T98N_N104Q",
    "MyoHead_120_E117S_S101Q",
    "MyoHead_120_E117S_L107R",
    "MyoHead_120_T98D_None",
    "MyoHead_120_T98N_L107R",
    "MyoHead_120_T98S_E117S",
    "MyoHead_120_N104K_None",
    "MyoHead_120_T98S_N104K",
    "MyoHead_120_E117S_N104Q",
    "MyoHead_120_None_None",
    "MyoHead_120_S101Q_L107R",
    "MyoHead_120_E117S_T98D",
    "MyoHead_120_S101Q_None",
    "MyoHead_120_N104Q_None",
    "MyoHead_120_N104K_T98D",
]
wait_time = 10

from pathlib import Path


def list_files(directory):
    p = Path(directory)
    return [str(file) for file in p.rglob("*") if file.is_file()]


# Example usage
files = list_files(directory)
files = [file for file in files if "_nanobody" in file]
for file in files:
    target_pdb = file[
        file.find("Canonacalized_") + len("Canonacalized_") : file.find("_nanobody.pdb")
    ]
    if target_pdb in FINISHED_RUNS:
        print("skipping target_pdb", target_pdb)
        continue
    # Navigate to the HADDOCK submission page
    driver.get("https://wenmr.science.uu.nl/haddock2.4/submit/1")

    # LOGIN time!
    # Wait for the page to load

    if first_run:
        wait = WebDriverWait(driver, 10)
        # Open the login dropdown
        login_dropdown_button = wait.until(
            EC.element_to_be_clickable(
                (By.CSS_SELECTOR, "a.dropdown-toggle.nav-link.active")
            )
        )
        login_dropdown_button.click()

        # Wait for the login form to become visible
        wait.until(EC.visibility_of_element_located((By.ID, "login_dropdown")))

        # Fill in the login credentials
        email_input = driver.find_element(By.ID, "email_dropdown")
        password_input = driver.find_element(By.ID, "pw_dropdown")

        # Replace these with your actual credentials
        email_input.send_keys(user_email)
        password_input.send_keys(user_password)
        password_input.send_keys(Keys.ENTER)

    job_name = target_pdb
    driver.find_element(By.ID, "runname").send_keys(job_name)

    # Set number of molecules to 2
    select_molecules = Select(driver.find_element(By.ID, "nb_partners"))
    select_molecules.select_by_visible_text("2")

    # Upload PDB file - ensure the path to the file is correct
    driver.find_element(By.ID, "p1_pdb_file").send_keys(file)
    driver.find_element(By.ID, "p2_pdb_file").send_keys(
        file.replace("nanobody.pdb", "myosinhaed.pdb")
    )

    if first_run:
        # Handle the cookie pop up
        wait = WebDriverWait(driver, 10)  # Increase the timeout as necessary
        cookie_consent_button = wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "a.cc-btn.cc-dismiss"))
        )

        # Click the "Got it!" button to dismiss the cookie consent
        cookie_consent_button.click()

    first_run = False

    if job_name in FINISHED_RUNS:
        continue

    wait = WebDriverWait(driver, 10)  # Increase the timeout as necessary

    # want to make it coarse grained!
    checkbox = driver.find_element(By.ID, "p1_cg")
    driver.execute_script("arguments[0].checked = true;", checkbox)
    checkbox = driver.find_element(By.ID, "p2_cg")
    driver.execute_script("arguments[0].checked = true;", checkbox)

    # Now that the cookie consent is handled, you can proceed with other actions, like clicking the "Next" button
    next_button = wait.until(EC.element_to_be_clickable((By.ID, "submit")))
    wait = WebDriverWait(driver, 10)  # Increase the timeout as necessary

    time.sleep(wait_time)
    next_button.click()

    wait = WebDriverWait(driver, 10)
    # Next set the residues
    input_field = driver.find_element(By.ID, "p1_r_activereslist_1")
    input_field.clear()  # Clear any existing content in the field
    input_field.send_keys(NANOBODY_CDH_RESIDUES)
    input_field = driver.find_element(By.ID, "p2_r_activereslist_1")
    input_field.clear()  # Clear any existing content in the field
    input_field.send_keys(IDX206_MYOSIN_HOTSPOTS)
    # input_field = driver.find_element(By.ID, "p3_r_activereslist_1")
    # input_field.clear()  # Clear any existing content in the field
    # input_field.send_keys(hotspots)
    # Empty out all the residues we don't care about
    # wait = WebDriverWait(driver, 10)
    # remove_button = wait.until(
    #     EC.element_to_be_clickable(
    #         (By.XPATH, '//a[@onclick="hideSelection(1, 2);"]')
    #     )
    # )
    # action = ActionChains(driver)
    # action.move_to_element(remove_button).perform()
    # remove_button.click()
    # # wait = WebDriverWait(driver, 10)
    # remove_button = wait.until(
    #     EC.element_to_be_clickable(
    #         (By.XPATH, '//a[@onclick="hideSelection(2, 2);"]')
    #     )
    # )
    # action = ActionChains(driver)
    # action.move_to_element(remove_button).perform()
    # remove_button.click()

    # wait = WebDriverWait(driver, 10)
    # remove_button = wait.until(
    #     EC.element_to_be_clickable(
    #         (By.XPATH, '//a[@onclick="hideSelection(3, 2);"]')
    #     )
    # )
    # action = ActionChains(driver)
    # action.move_to_element(remove_button).perform()
    # remove_button.click()

    # Get thru the next screen
    button = driver.find_element(By.ID, "submit")
    button.click()

    button = driver.find_element(By.ID, "submit")
    button.click()

    # Wait for the modal to be visible
    WebDriverWait(driver, 10).until(
        EC.visibility_of_element_located((By.ID, "covid-confirm"))
    )

    # Locate the "No" button and click it
    no_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.ID, "covidConfirmNo"))
    )
    no_button.click()

    wait = WebDriverWait(driver, 10)
    time.sleep(wait_time)
    current_url = driver.current_url
    print("collected URL: ", current_url)
    URLS_TO_CHECK_UP_ON.append(current_url)

    # Open a file for writing
    with open("HADDOCK_progress_urls.csv", "w", newline="") as file:
        writer = csv.writer(file)

        # Write each string in a separate row
        for item in URLS_TO_CHECK_UP_ON:
            writer.writerow([item])


# Close the driver after the automation process is complete
driver.quit()
