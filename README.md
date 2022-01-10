# Target contrastive pessimistic risk

This repository accompanies the paper:

"Robust domain-adaptive discriminant analysis"

published in Pattern Recognition Letters, vol. 248, pp 107-113, 2021 ([doi](https://doi.org/10.1016/j.patrec.2021.05.005)). It contains the algorithms and experimental protocol used to execute the experiments.

## Installation

Download:

- Junfeng Wens's Robust Covariate Shift Adjustment: https://webdocs.cs.ualberta.ca/~jwen4/codes/RobustLearning.zip
- Mark Schmidt's minFunc: http://www.cs.ubc.ca/~schmidtm/Software/minFunc.html

Add the unzipped folders to your path.

## Usage

Each folder marked __experiment\*__, contains a script starting with __run_exp\*__. It calls a function that contains experimental parameters, such as which classifiers to test, and runs the experiment. Results will be stored in a new folder. These can be gathered and printed by using the function __gather_exp\*__.

### Contact

Questions, bugs, and general feedback can be submitted to the [issues tracker](https://github.com/wmkouw/tcpr/issues).
