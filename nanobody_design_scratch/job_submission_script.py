# pip install selenium
# https://wenmr.science.uu.nl/haddock2.4/submit/1

# Chrome version 123.0.6312.87
# chrome://settings/help

# consulting this JSON file to try and find a suitable chromedriver
# https://googlechromelabs.github.io/chrome-for-testing/123.0.6312.0.json

# which says to download this chromedriver:
# https://storage.googleapis.com/chrome-for-testing-public/123.0.6312.0/mac-x64/chromedriver-mac-x64.zip


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

ROOT = "/Users/yitongtseo/Documents/GitHub/ethical_necromancy/nanobody_design_scratch/pdb_files"
NANOBODY_PATH = f"{ROOT}/canonacilized_3g9a_nanobody_VHH.pdb"
NANOBODY_CDH_RESIDUES = "26,27,28,29,30,31,32,33,51,52,53,54,55,56,57,58,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120"
URLS_TO_CHECK_UP_ON = []


def calculate_hotspot_idxs(
    pdb_name,
    hotspot_bucket,  # 0, 1, 2
    # Number of hotspot chunks (aka how many pieces do we split the protein into)
    chunk_num=3,
    # How many residues from the N & C terminus do we give as buffer
    end_buffer=10,
):
    if "rabbit_myosin" in pdb_name:
        protein_len = 100
        first_residue_idx = 0
    elif "2fxm" in pdb_name:
        # Chain A starts at residue 838, ends at residue 963, len = 125
        # Chain B starts at 850 and eds at 961
        # (Chains are still lined up) so let's go with chain B
        protein_len = 111
        first_residue_idx = 850
    elif "2fxo" in pdb_name:
        # Chain B of 2fxo_1 starts at residue 838, ends at residue 961, len = 123
        protein_len = 123
        first_residue_idx = 838

    final_residue_idx = first_residue_idx + protein_len
    first_residue_idx += end_buffer
    final_residue_idx -= end_buffer
    step_size = int((protein_len - (2 * end_buffer)) / chunk_num)

    hotspot_start_idx = (hotspot_bucket * step_size) + first_residue_idx
    hotspot_end_idx = min(
        ((hotspot_bucket + 1) * step_size) + first_residue_idx, final_residue_idx
    )
    return ",".join([f"{idx}" for idx in range(hotspot_start_idx, hotspot_end_idx)])


COILED_COIL_ACTIVE_RESIDUE_BUCKETS = {
    "hotspot_0": 0,
    "hotspot_1": 1,
    "hotspot_2": 2,
}

FINISHED_RUNS = [
    "myosin_binder_2fxm_hotspot_0",
    "myosin_binder_2fxo_1_hotspot_0",
    "myosin_binder_2fxo_2_hotspot_0",
    "myosin_binder_rabbit_0to100_hotspot_0",
    "myosin_binder_rabbit_50to150_hotspot_0",
    "myosin_binder_rabbit_100to200_hotspot_0",
    "myosin_binder_rabbit_150to250_hotspot_0",
    "myosin_binder_rabbit_200to300_hotspot_0",
    "myosin_binder_rabbit_250to350_hotspot_0",
]

wait_time = 1
files = [
    f"{ROOT}/canonicalized/2fxm",
    f"{ROOT}/canonicalized/2fxo_1",
    f"{ROOT}/canonicalized/2fxo_2",
    f"{ROOT}/canonicalized/relaxedrabbit_myosin_0to100",
    f"{ROOT}/canonicalized/relaxedrabbit_myosin_50to150",
    f"{ROOT}/canonicalized/relaxedrabbit_myosin_100to200",
    f"{ROOT}/canonicalized/relaxedrabbit_myosin_150to250",
    f"{ROOT}/canonicalized/relaxedrabbit_myosin_200to300",
    f"{ROOT}/canonicalized/relaxedrabbit_myosin_250to350",
    f"{ROOT}/canonicalized/relaxedrabbit_myosin_300to400",
    f"{ROOT}/canonicalized/relaxedrabbit_myosin_350to450",
    f"{ROOT}/canonicalized/relaxedrabbit_myosin_400to500",
    f"{ROOT}/canonicalized/relaxedrabbit_myosin_450to550",
    f"{ROOT}/canonicalized/relaxedrabbit_myosin_500to600",
    f"{ROOT}/canonicalized/relaxedrabbit_myosin_550to650",
    f"{ROOT}/canonicalized/relaxedrabbit_myosin_600to700",
    f"{ROOT}/canonicalized/relaxedrabbit_myosin_650to750",
    f"{ROOT}/canonicalized/relaxedrabbit_myosin_700to800",
    f"{ROOT}/canonicalized/relaxedrabbit_myosin_750to850",
    f"{ROOT}/canonicalized/relaxedrabbit_myosin_800to900",
    f"{ROOT}/canonicalized/relaxedrabbit_myosin_850to950",
    f"{ROOT}/canonicalized/relaxedrabbit_myosin_900to1000",
    f"{ROOT}/canonicalized/relaxedrabbit_myosin_950to1050",
    f"{ROOT}/canonicalized/relaxedrabbit_myosin_1000to1077",
]
user_email = input("What's your email yitong: ")
user_password = getpass.getpass("What's your password: ")

chromedriver_path = "/Users/yitongtseo/Documents/GitHub/ethical_necromancy/nanobody_design_scratch/chromedriver-mac-x64/chromedriver"
service = Service(executable_path=chromedriver_path)
driver = webdriver.Chrome(service=service)
first_run = True


for target_hotspot, hotspot_idx in COILED_COIL_ACTIVE_RESIDUE_BUCKETS.items():

    for test_coiled_coil_path in files:
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

        hotspots = calculate_hotspot_idxs(test_coiled_coil_path, hotspot_idx)
        target_pdb = (
            test_coiled_coil_path.split("/")[-1]
            .replace(".pdb", "")
            .replace("relaxedrabbit_myosin", "rabbit")
        )
        job_name = f"myosin_binder_{target_pdb}_{target_hotspot}"
        driver.find_element(By.ID, "runname").send_keys(job_name)

        # Set number of molecules to 3
        select_molecules = Select(driver.find_element(By.ID, "nb_partners"))
        select_molecules.select_by_visible_text("3")

        # Upload PDB file - ensure the path to the file is correct
        driver.find_element(By.ID, "p1_pdb_file").send_keys(NANOBODY_PATH)

        driver.find_element(By.ID, "p2_pdb_file").send_keys(
            f"{test_coiled_coil_path}_chainA.pdb"
        )
        driver.find_element(By.ID, "p3_pdb_file").send_keys(
            f"{test_coiled_coil_path}_chainB.pdb"
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
        input_field.send_keys(hotspots)
        input_field = driver.find_element(By.ID, "p3_r_activereslist_1")
        input_field.clear()  # Clear any existing content in the field
        input_field.send_keys(hotspots)

        # Empty out all the residues we don't care about
        wait = WebDriverWait(driver, 10)
        remove_button = wait.until(
            EC.element_to_be_clickable(
                (By.XPATH, '//a[@onclick="hideSelection(1, 2);"]')
            )
        )
        action = ActionChains(driver)
        action.move_to_element(remove_button).perform()
        remove_button.click()
        wait = WebDriverWait(driver, 10)
        remove_button = wait.until(
            EC.element_to_be_clickable(
                (By.XPATH, '//a[@onclick="hideSelection(2, 2);"]')
            )
        )
        action = ActionChains(driver)
        action.move_to_element(remove_button).perform()
        remove_button.click()
        wait = WebDriverWait(driver, 10)
        remove_button = wait.until(
            EC.element_to_be_clickable(
                (By.XPATH, '//a[@onclick="hideSelection(3, 2);"]')
            )
        )
        action = ActionChains(driver)
        action.move_to_element(remove_button).perform()
        remove_button.click()

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
