# Target contrastive pessimistic risk

This repository contains code of experiments in the paper:

"Robust domain-adaptive discriminant analysis"

which is currently under review.

This repository archives the implementations used to execute the experiments in the paper. For an up-to-date, easy-to-use implementation of TCPR, please refer to [libTLDA](https://github.com/wmkouw/libTLDA).

## Installation

Installation consists of downloading and adding to your path the following two packages:

- Junfeng Wens's Robust Covariate Shift Adjustment: https://webdocs.cs.ualberta.ca/~jwen4/codes/RobustLearning.zip
- Mark Schmidt's minFunc: http://www.cs.ubc.ca/~schmidtm/Software/minFunc.html

## Usage

Each folder marked __experiment\*__, contains a script starting with __run_exp\*__. It calls a function that contains experimental parameters, such as which classifiers to test, and runs the experiment. Results will be stored in a new folder. These can be gathered and printed by using the function __gather_exp\*__.

### Contact

Questions, bugs, and general feedback can be submitted to the [issues tracker](https://github.com/wmkouw/tcpr/issues).