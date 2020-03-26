#paths
project_name = 'Organoid-Image-Analysis'

base_path = "/home/pwahle/Organoid-Image-Analysis"

images_path = "/links/groups/treutlein/DATA/imaging/PW/4i"

modules_path = base_path + '/modules'

integration_path = scripts_path + '/integration'

functions_path = scripts_path + '/functions'

accessory_data_path = scripts_path + '/accessory_data'

results_path = base_path + '/resullts'

interim_results_path = base_path + '/interim_results'

images_pw_path = "/home/pwahle/images"

def stitched_images_path(plate = '/plate6', cycle = '/cycle1'):
    return images_path + plate + cycle + "/stitched"

def raw_images_path(plate = '/plate6', cycle = '/cycle1'):
    return images_path + plate + cycle + "/raw"

def zproject_images_path(plate = '/plate6', cycle = '/cycle1'):
    return images_path + plate + cycle + "/zproject"

def aligned_images_path(plate = '/plate6'):
    return images_path + plate + "/AlignedOrganoids" 



