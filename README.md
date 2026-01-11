Download the datasets and put them in data/ folder.

for ASSISTments 2009 download the skill builder dataset from https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010 and extract the skill builder csv in the data/assist2009 folder. 

for assistments 2012 download the data from https://sites.google.com/site/assistmentsdata/datasets/2012-13-school-data-with-affect and extract it in he data/assist2012 folder.

## Running Baselines (AKT)

We utilize the official implementation of [AKT (Attentive Knowledge Tracing)](https://github.com/arghosh/AKT).

### Prerequisites
Ensure your data is preprocessed using `match_dataset.py` and located in the `../data/` directory. The expected file structure is:
- `../data/assist2009/assist2009_akt_train.csv`
- `../data/assist2012/assist2012_akt_train.csv`

```bash
python3 match_dataset.py ../data/assist2012/2012-2013-data-with-predictions-4-final.csv
```

### Training Commands

To train the AKT model on **ASSISTments 2009**:
```bash
python main.py --dataset assist2009 --model akt_pid --dropout 0.05 --l2 1e-5 --batch_size 24 --d_model 256