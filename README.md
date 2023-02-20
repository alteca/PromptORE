# PromptORE – A Novel Approach Towards Fully Unsupervised Relation Extraction

Code for the CIKM'22 paper [PromptORE – A Novel Approach Towards Fully Unsupervised Relation Extraction](https://doi.org/10.1145/3511808.3557422).

We hope PromptORE will participate in improving Unsupervised Relation Extraction.

## Introduction

Unsupervised Relation Extraction (RE) aims to identify relations between entities in text, without having access to labeled data during training. This setting is particularly relevant for domain specific RE where no annotated dataset is available and for open-domain RE where the types of relations are *a priori* unknown.

Although recent approaches achieve promising results, they heavily depend on hyperparameters whose tuning would most often require labeled data. To mitigate the reliance on hyperparameters, we propose **PromptORE**, a "Prompt-based Open Relation Extraction" model. We adapt the novel prompt-tuning paradigm to work in an unsupervised setting, and use it to embed sentences expressing a relation. We then cluster these embeddings to discover candidate relations, and we experiment different strategies to automatically estimate an adequate number of clusters. To the best of our knowledge, PromptORE is the first unsupervised RE model that does not need hyperparameter tuning.

Results on three general and specific domain datasets show that PromptORE consistently outperforms state-of-the-art models with a relative gain of more than 40% in B3, V-measure and ARI. Qualitative analysis also indicates PromptORE’s ability to identify semantically coherent clusters that are very close to true relations.

## Installation

We have tested the installation with Python 3.8.16.

Install the packages listed in `requirements.txt`:

```bash
python3 -m pip install -r requirements.txt
```

## Running

The source code is specifically designed to work with the FewRel dataset [[1]](#cite-1) [[2]](#cite-2). To have more details on FewRel, please refer to <https://github.com/thunlp/FewRel>.

### Command Line Interface

PromptORE has the following parameters:
* `--seed=[SEED]`. Random state.
* `--n-rel=[K]`. Number of cluster, if the user knows it in advance.
* `--auto-n-rel`. Activates the estimation of the number of clusters using the elbow rule. *Mutually exclusive with `--n-rel`*.
* `--min-n-rel=[K]`. Only if `--auto-n-rel` is activated. Minimum number of clusters to test.
* `--max-n-rel=[K]`. Only if `--auto-n-rel` is activated. Maximum number of clusters to test.
* `--step-n-rel=[K]`. Only if `--auto-n-rel` is activated. Step to test clusters.
* `--max-len=[LEN]`. Maximum number of tokens in the instances (reasonable values are `fewrel=150, fewrel_nyt=500, fewrel_pubmed=250`).
* `files [FILE1] [FILE2] ...`. FewRel files to load for evaluation. All the files will be concatenated and the metrics aggregated.

### Clustering knowing *k*

For FewRel
```bash
python3 promptore.py --seed=0 --n-rel=80 --max-len=150 --files "<path-to-fewrel>/train_wiki.json" "<path-to-fewrel>/val_wiki.json"
```

For FewRel NYT
```bash
python3 promptore.py --seed=0 --n-rel=25 --max-len=500 --files "<path-to-fewrel>/val_nyt.json"
```

For FewRel PubMed
```bash
python3 promptore.py --seed=0 --n-rel=10 --max-len=250 --files "<path-to-fewrel>/val_pubmed.json"
```

### Estimating the number of clusters with the Elbow Rule

For FewRel
```bash
python3 promptore.py --seed=0 --auto-n-rel --min-n-rel=10 --max-n-rel=300 --step-n-rel=5 --max-len=150 --files "<path-to-fewrel>/train_wiki.json" "<path-to-fewrel>/val_wiki.json"
```

For FewRel NYT
```bash
python3 promptore.py --seed=0 --auto-n-rel --min-n-rel=2 --max-n-rel=100 --step-n-rel=2 --max-len=500 --files "<path-to-fewrel>/val_nyt.json"
```

For FewRel PubMed
```bash
python3 promptore.py --seed=0 --auto-n-rel --min-n-rel=2 --max-n-rel=100 --step-n-rel=2 --max-len=250 --files "<path-to-fewrel>/val_pubmed.json"
```

## License

The source code of PromptORE is licensed under the GPLv3 License. For more details, please refer to the [LICENSE.md file](LICENSE.md).

```
PromptORE
Copyright (C) 2022-2023 Alteca.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
```

## Contact

If you have questions using PromptORE, please e-mail us at pygenest@alteca.fr.

## Citation

If you make use of this code in your work, please kindly cite the following paper:

<div class="csl-entry">Genest, Pierre-Yves, Pierre-Edouard Portier, Elöd Egyed-Zsigmond, and Laurent-Walter Goix. “PromptORE - A Novel Approach Towards Fully Unsupervised Relation Extraction.” In <i>Proceedings of the 31st ACM International Conference on Information and Knowledge Management</i>, 11. Atlanta, USA: ACM, 2022. <a href="https://doi.org/10.1145/3511808.3557422">https://doi.org/10.1145/3511808.3557422</a>.</div>

<br/>

```bibtex
@inproceedings{10.1145/3511808.3557422,
    author = {Genest, Pierre-Yves and Portier, Pierre-Edouard and Egyed-Zsigmond, El\"{o}d and Goix, Laurent-Walter},
    title = {PromptORE - A Novel Approach Towards Fully Unsupervised Relation Extraction},
    year = {2022},
    isbn = {9781450392365},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3511808.3557422},
    doi = {10.1145/3511808.3557422},
    booktitle = {Proceedings of the 31st ACM International Conference on Information &amp; Knowledge Management},
    pages = {561–571},
    numpages = {11},
    location = {Atlanta, GA, USA},
    series = {CIKM '22}
}
```
  
## References

<div class="csl-entry"><a name="cite-1"></a><b>[1]</b> Han, Xu, Hao Zhu, Pengfei Yu, Ziyun Wang, Yuan Yao, Zhiyuan Liu, and Maosong Sun. “Fewrel: A Large-Scale Supervised Few-Shot Relation Classification Dataset with State-of-the-Art Evaluation.” In <i>Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing</i>, 4803–9. Brussels, Belgium: Association for Computational Linguistics, 2018. <a href="https://doi.org/10.18653/v1/d18-1514">https://doi.org/10.18653/v1/d18-1514</a>.</div>

<div class="csl-entry"><a name="cite-2"></a><b>[2]</b> Gao, Tianyu, Xu Han, Hao Zhu, Zhiyuan Liu, Peng Li, Maosong Sun, and Jie Zhou. “Fewrel 2.0: Towards More Challenging Few-Shot Relation Classification.” In <i>Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and 9th International Joint Conference on Natural Language Processing</i>, 6250–55. Hong Kong, China: Association for Computational Linguistics, 2019. <a href="https://doi.org/10.18653/v1/d19-1649">https://doi.org/10.18653/v1/d19-1649</a>.</div>
  