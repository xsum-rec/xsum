# xsum summarizing path explanations

This repository contains scripts for summarizing path-based explanations for recommendations.

## Contents

- **Scripts:** Python scripts for processing and converting data files.
- **Baselines**: The baseline ```pgpr_item_paths.jsonl``` and ```cafe_item_paths.jsonl``` can be found at the work of Balloccu et al. [ECIR 2024 Explainable Recommender Systems](https://github.com/explainablerecsys/ecir2024).

## Files

The repository includes the following Python scripts:

1. `xsum_item_group_pcst.py`
2. `xsum_item_group_steiner.py`
3. `xsum_item_pcst.py`
4. `xsum_item_steiner.py`
5. `xsum_user_group_pcst.py`
6. `xsum_user_group_steiner.py`
7. `xsum_user_pcst.py`
8. `xsum_user_steiner.py`

## Prerequisites

The scripts use standard Python libraries which can be installed via pip.

## Usage

Each script generates the explanation summarizations for users, items, user groups, and item groups using the Steiner and Prize Collecting Steiner Tree algorithms.
More details can be found in the paper "Summarizing path-based explanations", 2024

### Installation

Clone this repository to your local machine:

```bash
git clone https://github.com/yourusername/xsum.git
cd xsum




