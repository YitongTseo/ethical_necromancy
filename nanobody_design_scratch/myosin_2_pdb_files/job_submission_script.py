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

ROOT = "/Users/yitongtseo/Documents/GitHub/ethical_necromancy/nanobody_design_scratch/myosin_2_pdb_files"
NANOBODY_PATH = f"{ROOT}/canonacilized_3g9a_nanobody_VHH.pdb"
NANOBODY_CDH_RESIDUES = "26,27,28,29,30,31,32,33,51,52,53,54,55,56,57,58,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120"
NANOBODY_CDH2_AND_CDH3_RESIDUES = "51,52,53,54,55,56,57,58,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120"

URLS_TO_CHECK_UP_ON = []

import os
import sys
import urllib.request


import Bio
import Bio.PDB
import Bio.SeqRecord


def read_pdb(pdbcode, pdbfilenm):
    """
    Read a PDB structure from a file.
    :param pdbcode: A PDB ID string
    :param pdbfilenm: The PDB file
    :return: a Bio.PDB.Structure object or None if something went wrong
    """
    try:
        pdbparser = Bio.PDB.PDBParser(QUIET=True)  # suppress PDBConstructionWarning
        struct = pdbparser.get_structure(pdbcode, pdbfilenm)
        return struct
    except Exception as err:
        print(str(err), file=sys.stderr)
        return None


def get_hotspot(pdb):
    SELECTED_RESIDUES = "_".join(pdb.split("_")[:-1]) + "_selected_residues.pdb"
    selected_resi_pdb = read_pdb("selected_residues", SELECTED_RESIDUES)
    target_residues = set(selected_resi_pdb.get_residues())

    chunk_pdb = read_pdb("selected_residues", pdb)
    chunk_residues = set(chunk_pdb.get_residues())
    feature_residues = [
        r for r in target_residues.intersection(chunk_residues) if not r.is_disordered()
    ]

    print(
        "OVERLAP of ",
        str(len(target_residues)),
        " selected residues and "
        + str(len(chunk_residues))
        + " chunk size, is: "
        + str(len(feature_residues)),
    )
    return ','.join(sorted(
        ([str(residue.id[1]) for residue in target_residues.intersection(chunk_residues)])
    ))


FINISHED_RUNS = []

wait_time = 1
files = (
    [f"{ROOT}/5h53_chunks/5h53_chunk{idx}.pdb" for idx in range(1, 11)]
    + [f"{ROOT}/6bih_chunks/6bih_chunk{idx}.pdb" for idx in range(1, 11)]
    + [f"{ROOT}/6ysy_chunks/6ysy_chunk{idx}.pdb" for idx in range(1, 11)]
    + [f"{ROOT}/8efd_chunks/8efd_chunk{idx}.pdb" for idx in range(1, 11)]
)

user_email = input("What's your email yitong: ")
user_password = getpass.getpass("What's your password: ")

chromedriver_path = "/Users/yitongtseo/Documents/GitHub/ethical_necromancy/nanobody_design_scratch/chromedriver-mac-x64/chromedriver"
service = Service(executable_path=chromedriver_path)
driver = webdriver.Chrome(service=service)
first_run = True

for myosin_chunk in files:
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

    hotspots = get_hotspot(myosin_chunk)
    print('YITONG hotspots', hotspots)
    target_pdb = myosin_chunk.split("/")[-1].replace(".pdb", "")
    job_name = target_pdb
    driver.find_element(By.ID, "runname").send_keys(job_name)

    # Set number of molecules to 2
    select_molecules = Select(driver.find_element(By.ID, "nb_partners"))
    select_molecules.select_by_visible_text("2")

    # Upload PDB file - ensure the path to the file is correct
    driver.find_element(By.ID, "p1_pdb_file").send_keys(NANOBODY_PATH)

    driver.find_element(By.ID, "p2_pdb_file").send_keys(myosin_chunk)
    # driver.find_element(By.ID, "p3_pdb_file").send_keys(
    #     f"{test_coiled_coil_path}_chainB.pdb"
    # )

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
    input_field.send_keys(NANOBODY_CDH2_AND_CDH3_RESIDUES)
    input_field = driver.find_element(By.ID, "p2_r_activereslist_1")
    input_field.clear()  # Clear any existing content in the field
    input_field.send_keys(hotspots)
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

    try:
        # Wait for the alert to appear
        alert = WebDriverWait(driver, 10).until(EC.alert_is_present())

        # Print the alert text (optional)
        print("Alert text:", alert.text)

        # Dismiss the alert
        alert.dismiss()
    except:
        pass

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
