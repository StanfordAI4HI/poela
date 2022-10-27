### Instructions for running the code

To run POELA:

```bash
python src/run_pg.py --action_mask_type=nn_action_dist --threshold=0.6 --var_coeff=0.1
```

To run PO-mu/PO-CRM:

```bash
python src/run_pg.py --action_mask_type=step --threshold=0.01 --var_coeff=10.0
```

To run BCQ:
```bash
python src/run_ql.py --state_clipping=0 --threshold=0.01
```


To run PQL:
```bash
python src/run_ql.py --state_clipping=1 --threshold=0.01
```

### Acknowledgements
The code is an adaptation of the BCQ official [implementation](https://github.com/sfujim/BCQ).

This is code for the paper Offline Policy Optimization with Eligible Actions 
https://arxiv.org/abs/2207.00632
Yao Liu, Yannis Flet-Berliac, and Emma Brunskill 
Conference on Uncertainty in AI (UAI) 2022
