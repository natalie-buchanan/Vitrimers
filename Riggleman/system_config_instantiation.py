from system_config_functions import write_system_config
import sys

"System parameters: Networks with 1:1 stoichiometry between reactive groups"

nu_list = [5] # Number of segments per polymer chain
phi_list = [0.25] # Mole fraction of polymer particles (segments plus cross-linkers) in melt
num_nu_list = [20000] # [80000, 10000] # Number of polymer segments
config_list = ['003'] # ['001', '002'] # List of system configuration identifiers

f = 4 # Functionality of cross-linkers
b = 1.0 # Kuhn length

if len(config_list) != len(nu_list) * len(phi_list) * len(num_nu_list):
    sys.exit("The explicit number of system configurations identifiers must equal the total number of system configurations.")

config_indx = 0

for nu in nu_list:
    for phi in phi_list:
        for num_nu in num_nu_list:
            num_clinkr = int(num_nu / (2 * nu)) # Number of cross-linkers if stoichiometric ratio = 1.0
            num_plymr_prtcl = num_nu + num_clinkr # Number of polymer particles (segments plus cross-linkers)
            num_systm_prtcl = int(num_plymr_prtcl / phi) # Number of system particles

            filename = "./system-configurations/" + config_list[config_indx] + ".config"
            
            write_system_config(filename, nu, phi, num_nu, f, b, excess_linker=1.0)

            config_indx += 1