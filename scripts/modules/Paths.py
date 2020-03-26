
project_name = 'Organoid-Image-Analysis'

#paths
base_path = "/home/pwahle/Organoid-Image-Analysis" #hopefully this will remain the only place with <username>, exchange accordningly

images_path = "/links/groups/treutlein/DATA/imaging/PW/4i"

modules_path = base_path + '/modules'

integration_path = scripts_path + '/integration'

functions_path = scripts_path + '/functions'

accessory_data_path = scripts_path + '/accessory_data'

results_path = base_path + '/results'

interim_results_path = base_path + '/interim_results'

images_pw_path = "/home/pwahle/images" #this is only for pw

def stitched_images_path(plate = '/plate6', cycle = '/cycle1'):
    return images_path + plate + cycle + "/stitched"

def raw_images_path(plate = '/plate6', cycle = '/cycle1'):
    return images_path + plate + cycle + "/raw"

def zproject_images_path(plate = '/plate6', cycle = '/cycle1'):
    return images_path + plate + cycle + "/zproject"

def aligned_images_path(plate = '/plate6'):
    return images_path + plate + "/AlignedOrganoids" 



