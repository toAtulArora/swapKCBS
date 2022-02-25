# Numerics for Self-Testing of a Single Quantum System: Theory and Experiment

This folder contains the numerics for arXiv:xxx.xxx. 

Feel free to contact us for any details.

* [SDP_code](SDP_code): Contains our algorithm which generates an SDP to lower bound the total fidelity using symbolic computation ```sympy``` and solving the SDP using ```numpy``` and ```cvxpy``` via MOSEK (use SCS for faster, but possibly less reliable results).
* [All_data](All_data): Contains the experimental data.
* [Configuration](Configuration): KCBS configuration: optimal and sub-optimal. How close is a given configuration (in three dimensions) is to an ideal configuration, determined up to a unitary. This is done by parametrising unitaries in 3 dimension with eight parameters.
* [Repeatability](Repeatability) Contains a jupyter notebook which computes the repeatability of each measurement and the associated results.
* [Violations_and_orthogonality](Violations_and_orthogonality) KCBS value and deviation from orthogonality.
* [Robustness_curve](Robustness_curve) Contains the notebook we used to generate the main figure in the article.


## Authors

| Name | email |
|-|-|
| Atul Singh Arora | toAtulArora@gmail.com |
| Xiao-Min Hu | huxm@ustc.edu.cn |
| Kishor Bharti | kishor.bharti1@gmail.com |

<img width="1559" alt="image" src="https://user-images.githubusercontent.com/2003122/155685309-0ab4aa98-529a-4860-aadd-d0f3f72356b8.png">
