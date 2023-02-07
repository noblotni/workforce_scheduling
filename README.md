# Workforce scheduling

An algorithm to establish the schedule of a company.

## Installation

To use this package, you will need python 3.9 and the solver GUROBI. To install the package:

1. Clone the repository
```
git clone https://github.com/noblotni/workforce_scheduling
```
2. Go to the folder `workforce_scheduling`
```
cd workforce_scheduling
```
3. Install the dependencies
```
pip install .
``` 

## Usage of the command line interface (CLI)

You can read the documentation of the CLI using the command:
```shell
python -m workforce_scheduling --help
```
It displays the following message:
```
usage: Workforce scheduling [-h] {solve,classify} ...

positional arguments:
  {solve,classify}  sub-commands help
    solve           Solve the optimization problem on an instance.
    classify        Classify the solutions with a preference model.

optional arguments:
  -h, --help        show this help message and exit
```

### Usage of the solve subcommand
Type this command to display the help:
```shell
python -m workforce_scheduling solve --help
```
Help of the `solve` subcommand:
```
usage: workforce_scheduling solve [-h] --data-path DATA_PATH [--nb-processes NB_PROCESSES]
                                  [--gurobi-threads GUROBI_THREADS] [--output-folder OUTPUT_FOLDER]

optional arguments:
  -h, --help            show this help message and exit
  --data-path DATA_PATH
                        Path to the data file. Must be a json file.
  --nb-processes NB_PROCESSES
                        Number of processes for the solution search. (default: 1)
  --gurobi-threads GUROBI_THREADS
                        Maximal number of threads for Gurobi. (default: 4)
  --output-folder OUTPUT_FOLDER, -o OUTPUT_FOLDER
                        Folder where to save the output files. (default: ./solved/data_filename)
```
### Usage of the classify subcommand

Type this command to display the help:
```
python -m workforce_scheduling classify --help
```
Help of the `classify` subcommand:
```
usage: workforce_scheduling classify [-h] --pareto-path PARETO_PATH --preorder-path PREORDER_PATH
                                     [--pref-model PREF_MODEL]

optional arguments:
  -h, --help            show this help message and exit
  --pareto-path PARETO_PATH
                        Path to the calculated Pareto surface.
  --preorder-path PREORDER_PATH
                        Path to the preorder on a subset of solutions.
  --pref-model PREF_MODEL
                        Preferences model to use: UTA or k-best (default:UTA)
```