Provided casm project is stripped of all vasp calculation files. The vasp results are contained within files named 'properties.calc.json' within every supercell directory 


Steps:
--------

activate casm environment


cd to root casm directory (rocksalt_shared_ZrN_casm)


run casm update         #attempt to import vasp calculation data, if not already present


run casm bset -uf       #enumerate all basis functions. Have to do this when starting on a new machine, even if they already exist


create or cd to a ce_fits directory 


*ensure that only supercells with calculated energies are used for training. should already be fine.


the file 'genetic_alg_settings.json' contains all the input parameters required to run a genetic fit using casm-learn.
This uses the vasp calculation results contained in the casm_root/training_data/ as the "target" energy data, where the genetic algorithm selects and scales the "ECI" (effective cluster interaction) parameters to best reproduce the "target" vasp density functional theory (DFT) energies


We are often changing the "A", "B" and "kT" terms, which are found in the 'genetic_alg_settings.json' file. 
These parameters give us our fit penalty, where the penalty is calculated as: B + A*exp(E_fit - E_training)


To perform a fit, the 'genetic_alg_settings.json' file and 'training_set.txt' file must be in the same directory.
The fit can be performed by running:

casm-learn -s genetic_alg_settings.json > fit.out


Each genetic algorithm fit collects a specified handful (exact number is specified in the input json) of top fits, according to lowest CV and rms scores.
The best fit is given the label 0, second best is named 1, and so on. 
To output plottable data for one of these fits, run the command:

casm-learn -s genetic_alg_settings.json --checkhull --indiv 0 > check.0         
#where 0 corresponds to the fit with the lowest CV score. This can be done for other fits (i.e. replace 0 with 1,2,3,...)


To actually plot the data, we have provided a python script named 'plot_clex_hull_data.py'. This will use the output files from the above --checkhull command. 

This script can be run by:
python path/to/plot_clex_hull_data.py    /full/path/to/this/directory/  id_number_for_your_fit_of_interest

for example, assuming you are in the same directory as the fit data, plotting the data for the ECI fit with the lowest CV score is done by using the command:

python path/to/plot_clex_hull_data.py  \`pwd\` 0
