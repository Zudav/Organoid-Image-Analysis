import os
import fnmatch
import numpy as np
import sys
import shutil
from skimage import io

#Z:\DATA\imaging\PW\4i\plat6\plate6
def define_organoids():
    print("Specify the cycles and files you would like to align. The 'masterfolder' should contain all the cycles as subfolders.")
    print("If at some point you are unsure about what to enter try '?' for help")
    # Ask for path to folder containting all cycles as subfolders
    while True:
        masterfolder = input("Path to masterfolder: \n")
        if not os.path.isdir(masterfolder):
            print("Path: \n", masterfolder, "\nnot found")
            continue
        else:
            # Find all folders containing the word "cycle"
            cycle_folders = [f for f in os.listdir(masterfolder) if "cycle" in f]
            if len(cycle_folders) == 0:
                print("No subfolders containing the word 'cycle' were found. Please specify a valid path.")
                continue
            else:
                print("Found the following subfolders:\n", "\n ".join(cycle_folders))
            break
     
    # Find out which folders the user wants to align       
    while True:
        cycle_input = input("Specify cycles to be aligned. Can be a list of space-separated numbers (e.g. '1 2' for cycle1 and cycle2) or 'all':\n")
        if cycle_input == "?":
            folder_numbers = [s.strip("cycle") for s in cycle_folders]
            print("Enter any combination (no duplicates) of the cycles, separated by a space (e.g. '1 2') or simply 'all'")
            print("The following inputs are valid:\n", "\n ".join(folder_numbers))
        elif cycle_input == "all":
            # We don't need to change the already existing "cycle_folders" list
            relevant_cycles = cycle_folders
            break
        else:
            # Turn input into a list
            cycle_input = cycle_input.split()
            # Turn e.g. '1' into cycle1
            cycle_input = ["cycle" + number for number in cycle_input]
            missing_cycles = []
            # For each specified cycle check if it's contained in the list of folders
            for c in cycle_input:
                if c not in cycle_folders:
                    missing_cycles.append(c)
            if len(missing_cycles) > 0:
                print("The following folder(s) could not be found:\n", "\n ".join(missing_cycles))
                print("Enter '?' to see your options.")
                continue
            else:
                print("You picked the following folder(s):\n", "\n ".join(cycle_input))
                confirm = input("Continue with these folders? (y/n): ")
                if confirm.lower() == "y":
                    relevant_cycles = cycle_input
                    break
                elif confirm.lower() == "n":
                    continue
                else:
                    print("Input has to be either 'y' or 'n'")
                    continue
    
    
    # Get the reference cycle and put it to the front of the list
    while True:
        reference_cycle = input("Specify which cycle you want to use as the reference (e.g. '1' for cycle1):\n")
        # Get list of cycle numbers
        relevant_cycles_numbers = [s.strip("cycle") for s in relevant_cycles]
        if reference_cycle == "?":
            print("The following inputs are valid:\n", "\n ".join(relevant_cycles_numbers))
            continue
        else:
            reference_cycle = "cycle" + reference_cycle
            if reference_cycle not in relevant_cycles:
                print("Name\n", (reference_cycle), "\nnot found.")
                print("Enter '?' to see your options.")
                continue
            else:
                cycles_sorted = relevant_cycles
                # Move reference cycle to the beginning of the list
                cycles_sorted.insert(0, cycles_sorted.pop(cycles_sorted.index(reference_cycle)))
                break
        
    # Go to the reference cycle folder, open the "stitched" folder and display the filenames
    file_path = os.path.join(masterfolder, reference_cycle, "stitched")
    if not os.path.exists(file_path):
        print("Could not find the following directory: \n", file_path)
        sys.exit()
    organoid_files = fnmatch.filter(os.listdir(file_path), "*.tif")
    print("Found the following files:\n", "\n ".join(organoid_files))
    
    # Find out which files the user wants to align
    while True:
        file_input = input("Specify files to be aligned. Can be a list of space-separated indices (starting at 0) or 'all':\n")
        if file_input == "?":
            print("Enter a list of indices (e.g. '0 1 3 4') or 'all'")
            print("The following inputs are valid:\n", list(np.arange(len(organoid_files))))
            continue
        elif file_input == "all":
            # We don't need to change the already existing "organoid_files" list
            relevant_files = organoid_files
            break
        else:
            # Turn input into a list
            file_input = file_input.split()
            # Turn list of strings into list of int
            file_input = list(map(int, file_input))
            relevant_files = [organoid_files[i] for i in file_input]
            print("You picked the following files:\n", "\n ".join(relevant_files))
            confirm = input("Continue with these files? (y/n): ")
            if confirm.lower() == "y":
                break
            elif confirm.lower() == "n":
                continue
            else:
                print("Input has to be either 'y' or 'n'")
                continue
            
    # Check if every cycle contains the specified filenames
    missing_files = []
    for cycle in relevant_cycles:
        file_path = os.path.join(masterfolder, cycle, "stitched")
        organoid_files = os.listdir(file_path)
        for file in relevant_files:
            if file not in organoid_files:
                missing_files.append(os.path.join(cycle, file))
    
    if len(missing_files) == 0:
        print("Found the specified files in all relevant cycles")
    else:
        print("The following files could not be found:")
        for file in missing_files:
            print(file)
        sys.exit()
        
    return masterfolder, relevant_cycles, relevant_files

def save_images(path, organoid, cycles, cropped_images, corr_matrix, bin_matrix, overwrite=False):    
    # Create an "alignment" folder if it doesn't already exist
    alignment_folder = os.path.join(path, "AlignedOrganoids")
    if not os.path.exists(alignment_folder):
        os.makedirs(alignment_folder)
    # Remove the .tif file extension
    organoid = organoid.replace(".tif", "")
    # Check if a folder with the organoid name already exists
    organoid_folder = os.path.join(alignment_folder, organoid)
    if not os.path.exists(organoid_folder):
        os.makedirs(organoid_folder)
    else:
        if overwrite==True:
            # Remove directory and create it again, i.e. overwrite it
            shutil.rmtree(organoid_folder)
            os.makedirs(organoid_folder)
        else: # i.e. overwrite==False
            i = 1
            while True:
                # Create a new folder with the same name and a number at the end
                organoid_folder = os.path.join(alignment_folder, organoid+"_"+str(i))
                if not os.path.exists(organoid_folder):
                    os.makedirs(organoid_folder)
                    break
                else: # increase the number if e.g. organoid-name_1 already exists
                    i += 1
                    continue
    # Save all the images
    for i in range(len(cropped_images)):
        filename = cycles[i] + "_aligned.tif"
        img_path = os.path.join(organoid_folder, filename)
        io.imsave(img_path, cropped_images[i], check_contrast=False)
    
    # Create a text file and save additional information
    info = open(os.path.join(organoid_folder, "alignment_info.txt"), "w")
    info.write("Organoid: {}\n\n"
               "Cycles: {}\n\n"
               "Reference cycle: {}\n\n"
               "Correlation matrix (same order as cycles):\n"
               "{}\n\n"
               "Binary similarity matrix:\n"
               "{}".format(organoid, cycles, cycles[0], 
                corr_matrix, bin_matrix))
    info.close()

if __name__ == "__main__":
    masterfolder, relevant_cycles, relevant_files = define_organoids()















