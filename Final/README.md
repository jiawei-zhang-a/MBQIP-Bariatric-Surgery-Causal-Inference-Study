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

This study focuses on estimating the causal effect of bariatric surgery on various health outcomes using observational data from the MBQIP. There are several prominent procedures for bariatric surgery - Sleeve Gastrectomy, Roux-en-Y Gastric Bypass (RYGB), Adjustable Gastric Band (AGB), Biliopancreatic Diversion with Duodenal Switch (BPD/DS), and Single Anastomosis Duodeno-Ileal Bypass with Sleeve Gastrectomy (SADI-S). We consider the most widely used Sleeve as the baseline and study the effect and risk for other procedures from a causal view. We employ state-of-the-art average treatment effect (ATE) estimation methods -  causal forest and double machine learning from EconML, and rewrite the method - DragonNet by ourselves. We apply the methods above to MBQIP (Medicare Beneficiary Quality Improvement Project) datasets to study the change in BMI as the effects and the probability of death, intervention, readmission, and reoperation as the risks before and after the surgery. Meanwhile, we compare the risks among different procedures via the relative risk (RR), which gives a more exact result for classification. 



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
