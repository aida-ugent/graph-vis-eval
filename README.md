# Evaluating Graph Visualizations by Representation Learning and Graph Layout Methods #

This repository contains code to compare two-dimensional embeddings from graph representation learning methods and graph layout methods. We analyze the embeddings measuring readability (crosslessness, minimum-angle, edge-length variation, Gabriel shape), neighborhood preservation, stress, and link-prediction AUC. 


## Submodules ##

We include all evaluated methods as submodules in `methods/` and the evaluation tools in `tools/`.
Graph representation learning methods:
- [AROPE][1]
- [CNE][2] (original [CNE](https://bitbucket.org/ghentdatascience/cne/) repository)
- [Deepwalk][3]
- [GCN AE][6]

Graph layout methods:
- [DRGraph][4]
- [Fruchterman-Reingold][5]

Evaluation tools:
- [EvalNE][7]
- [GLAM][8]


## Installation and Setup ##

1. Clone the repository with all submodules.
    ```bash
    git clone --recursive URL
    ```
2. Install the approximate Fruchterman-Reingold in `methods/frrtx` following the instructions [here][5]. If you change the installation path it might be necessary to set the correct path to the executable in [config.ini](config.ini) and [evalne_config.ini](evalne_config.ini).

3. Install DRGraph in `methods/drgraph` following the instructions [here][4]. If you change the installation path it might be necessary to set the correct path to the executable in [config.ini](config.ini) and [evalne_config.ini](evalne_config.ini).

4. Install GLAM in `tools/glam` following the instructions [here][8] and set the correct path to the executable in [config.ini](config.ini) .
    ```bash
    [glam]
    method = glam_scores
    args = {'glam_path': 'tools/glam/build/glam'}
    ```

5. Install all packages in `requirements.txt` and version 0.3.3 of [EvalNE][7] in the same python 3.6 environment. We will run CNE, Fruchterman-Reingold, and DRGraph from this environment. 
    ```
    virtualenv -p /usr/bin/python3 gravis_venv
	gravis_venv/bin/pip3 install -r requirements.txt

    cd tools/evalne
    ../../gravis_venv/bin/python3 setup.py install
    ```
    Make sure Tkinter (`python3-tk`) is installed on your system as well. 

6. Create separate environments for AROPE, DeepWalk, and GAE (e.g. in the methods directory) with python 2.7. The following commands assume `/usr/bin/python` has version 2.7.
    ```bash AROPE
    virtualenv -p /usr/bin/python arope_venv
    arope_venv/bin/pip install -r arope_requirements.txt
    arope_venv/bin/python arope_setup.py install
    ```
    ```bash DeepWalk
    # ensure python-dev is installed and LD_LIBRARY_PATH is empty
    virtualenv -p /usr/bin/python deepwalk_venv
    deepwalk_venv/bin/pip install -r deepwalk_requirements.txt
    cd deepwalk
    ../deepwalk_venv/bin/python setup.py install
    ```
    ```bash GAE
    virtualenv -p /usr/bin/python gae_venv
	gae_venv/bin/pip install -r gae_requirements.txt
    cd gae
    ../gae_venv/bin/python setup.py install
    ```

## Usage ##

The output folder, the datasets, methods, and measures to evaluate can be easily changed in `config.init` and `evalne_config.ini`. 
To compute visualizations and the readability and distance-based measures run `main.py` in the new environment. 
```bash
source gravis_venv/bin/activate
python main.py
```
With the default settings, a new folder will be create in `/output` containing a subfolder with embeddings and visualizations for each network dataset, the config file used, and the results in `result.txt`. 

To compute the link-prediction performance, run `evalne_main.py` in the same python environment. 
```bash
source gravis_venv/bin/activate
python evalne_main.py
```
With the default settings, a new folder will be created in `/output` containing a subfolder with train/test edge splits for each dataset, the config file used, and the results in `lp_results.txt`. 


## License ##

This sample code is licensed under the MIT License (MIT)


[1]:    https://github.com/ZW-ZHANG/AROPE
[2]:    https://github.com/heitere/cne-gravis.git
[3]:    https://github.com/phanein/deepwalk
[4]:    https://github.com/ZJUVAI/DRGraph
[5]:    https://github.com/owl-project/owl-graph-drawing
[6]:    https://github.com/tkipf/gae
[7]:    https://github.com/Dru-Mara/EvalNE
[8]:    https://github.com/VIDILabs/glam