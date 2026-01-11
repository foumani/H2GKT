# H2GKT: Hybrid Heterogeneous Graph Framework for Knowledge Tracing

## Dataset Preparation
Download the datasets and put them in the `data/` folder.

* **ASSISTments 2009:** Download the 'skill builder' dataset from [here](https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010) and extract the skill builder CSV into the `data/assist2009` folder.
* **ASSISTments 2012:** Download the data from [here](https://sites.google.com/site/assistmentsdata/datasets/2012-13-school-data-with-affect) and extract it into the `data/assist2012` folder.

## Prerequisites & Installation
This code requires **Python 3.11**.

To install the necessary dependencies, run:
```bash
pip install -r requirements.txt
```
## Usage
To run the code, use the following command:
```bash
python3 main.py --dataset assist2009
```