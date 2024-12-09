MTBench: a homework

quick start:
1. download model from: https://pan.baidu.com/s/1DwfnLA1ayp4C6HcMT39Vtg?pwd=ke1m, with key：ke1m.
2. unzip it in ``/MTBench/results/final_model``
3. ``cd /MTBench/``
4. ``python quick_start.py``

how to eval a model:
1. use ``generate_output.py`` to generate outputs from the model to be eval,
   the outputs will be saved at ``/MTBench/eval_data/{model_name}.jsonl``.
2. use ``eval.py`` to eval the outputs by MTBench,
   the result will be saved at ``/MTBench/eval_result/{model_name}.jsonl``
   & the final result will be print in the console.

how to train MTBench:
1. check ``/MTBench/train.py`` for further infomation.
