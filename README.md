# Hierarchical multi-task learning, self-supervised learning, Self-organizing map

Seunghan Lee and Taeyong Park



### Step 1) Dataset Generation: `generate_dataset.py`

- Input files 
  - Metadata (Table) Data
  - Log Data 1)

- Output files
  - Preprocessed Data: `doll_merged.csv`
  - User data with Factor Analysis: `FA_doll_merged.csv`
  - Log Data (a) with Factor Analysis: `FA_log_action.csv`
  - Log Data (b) with Factor Analysis: `FA_log_program.csv`



Example

```bash
$ python generate_dataset.py --doll=6 --action=3 --program=4
```

- Number of factors for Factor Analysis: integrated data=6, Log Data (a)=3, Log Data (b)=4

<br>

### Step 2) Clustering: `clustering.py`

- Input files 
  - User data with Factor Analysis: `FA_doll_merged.csv`
  - Log Data (a) with Factor Analysis: `FA_log_action.csv`
  - Log Data (b) with Factor Analysis: `FA_log_program.csv`
- Output files
  - Clustered Customer Data: `FA_with_cluster.csv`
- Main Procedure Steps:
  - (1) Data merging
  - (2) Scaling
  - (3) Removing outlier customers
  - (4) Clustering

<br>

Example

```bash
$ python clustering.py --grid_x=2 --grid_y=3 --iter=20000
```

- SOM Cluster size: (2,3)
- Number of iterations for SOM training: 20,000

<br>

### Step 3) Task Labeling: `task_label.py`

- Input files 
  - Clustered Customer Data: `FA_with_cluster.csv`
- Output files
  - Clustered Customer Data with Task Labels: `cluster_df_merged.csv`

<br>

Example

```bash
$ python task_label.py
```

<br>

### Step 4) Emergency Call Prediction Modeling: `model_MTL.py`

Experiment Settings:

- Task: Binary Classification
- Threshold: Try 0.2, 0.4, 0.6, 0.8 and set the optimal boundary value.

<br>

Example

```bash
$ python model_MTL.py --epoch=200
```
