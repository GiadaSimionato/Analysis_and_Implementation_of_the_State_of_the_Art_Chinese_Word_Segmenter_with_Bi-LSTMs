
# Analysis and Implementation of the State-of-the-Art Chinese Word Segmenter with Bi-LSTMs
Project repository of the first homework for the course of Natural Language Processing, Sapienza University of Rome.

This repository contains the Python implementation of the framework for data preprocessing and deep Chinese Word Segmentation based on Bi-LSTMs networks.
The implementation was tested on the AS, CITYU, PKU and MSR datasets.

This work was completed on April 24th, 2019.

## Contents:

- [Structure](#structure)
- [Execution](#execution)
- [References](#references)

## Structure
- **code** folder contains the full code of the framework.
- **report.pdf** is the report for this project.

## Execution
1. Run the `main.py` file
2. For the required resouces, including the pretrained best model weights (i.e. bigram_model_train, char_model_train and modelBest.h5) please contact me at giada.simionato.3@gmail.com

## References

The papers used in this work include, but are not limited to:

- J. Zhou, J. Wang, G. Liu. *Multiple Character Embeddings for Chinese Word Segmentation.* 2019 Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pp. 210-216, doi: 10.18653/v1/P19-2029.
- Y. Li, W. Li, F. Sun, S. Li. *Component-Enhanced Chinese Character Embeddings.* Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, Lisbon, 2015, pp. 829–834, doi: 10.18653/v1/D15-1098.
- J. Yang, Y. Zhang, S. Liang. *Subword Encoding in Lattice LSTM for ChineseWord Segmentation.* Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics, Minneapolis, 2019, pp. 2720–2725, doi: 10.18653/v1/N19-1278.
- J. Ma, K. Ganchev, D. Weiss. *State-of-the-art ChineseWord Segmentation with Bi-LSTMs.*, Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, Brussels, 2018, pp. 4902–4908, doi: 10.18653/v1/D18-1529.
- Y. Kitagawa, M. Komachi. *Long Short-Term Memory for JapaneseWord Segmentation.*, Pacific Asia Conference on Language Information and Computation, 2018.
- W. Huang, X. Cheng, K. Chen, T. Wang, W. Chu. *Toward Fast and Accurate Neural ChineseWord Segmentation with Multi-Criteria Learning.* 2019. 
