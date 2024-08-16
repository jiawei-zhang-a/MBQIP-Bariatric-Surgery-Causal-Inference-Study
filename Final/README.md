# MBQIP Bariatric Surgery Causal Inference Study

This repository contains scripts and datasets for our paper 'Comparisons of the Treatment and Side Effects of Several Bariatric Surgery Procedures: An Observational Study via Random Forest-based and Neural Network-based Approaches'. In this work, we study the effect and risk of different bariatric surgery procedures from a causal inference view. 

## Table of Contents

- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributors](#contributors)
- [License](#license)

## Introduction

This study investigates the causal impact of bariatric surgery on various health outcomes using observational data from the MBQIP (Medicare Beneficiary Quality Improvement Project). Bariatric surgery encompasses several prominent procedures, including Sleeve Gastrectomy, Roux-en-Y Gastric Bypass (RYGB), Adjustable Gastric Band (AGB), Biliopancreatic Diversion with Duodenal Switch (BPD/DS), and Single Anastomosis Duodeno-Ileal Bypass with Sleeve Gastrectomy (SADI-S). In this analysis, we consider Sleeve Gastrectomy as the baseline procedure and evaluate the effects and risks associated with the other procedures from a causal perspective.

We employ state-of-the-art average treatment effect (ATE) estimation methods -  causal forest and double machine learning from EconML, and rewrite the method - DragonNet - in pytorch by ourselves. We apply these methods to the MBQIP datasets to estimate the average treatment effect (ATE) for changes in BMI before and after surgery as the variable to study the effects, while we estimate the relative risk (RR) for the probability of death, intervention, readmission, and reoperation to study the risks. 

We post our Python scripts to conduct the mentioned methods and the MBQIP datasets used in this work. 


## Dependencies

- Python 3.8+
- numpy
- pandas
- scikit-learn
- tensorflow
- keras
- econml
- dragonnet

## Installation

1. Clone this repository:

2. Set up a virtual environment and activate it:

3. Install the required dependencies:

`pip install -r requirements.txt`

## Usage

## Results

The results of the causal inference study, including estimated treatment effects and performance metrics, will be saved in the `results` directory. Additionally, detailed logs and plots will be provided for further evaluation.

## License

This project is licensed under the [MIT License](LICENSE.md).
