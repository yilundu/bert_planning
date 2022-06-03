# Planning with BERT
This is a simple implementation of planning with BERT model. 

To quickly train a bert model use
```
./train.sh
```

To then use the bert model for planning, run the following command

```
python test.py
```

The overall idea is that BERT defines an energy function over discrete
trajectories $E(s_0, s_1, \ldots, s_T) = \prod_i p(s_i|s_{\\i}$ -- we can run MCMC sampling over this energy function to generate
a trajectory.

We can then bias the MCMC sampling protocal on this energy function to generate different trajectories (BERT has a mouth and it can speak)
We want to see if we can modulate this trajectory energy function with different goal energy function -- such as for example that the goal has
to be a particular state, and see if MCMC sampling can successfully generate a correct state.



## Requirement
- python==3.6
- torch==1.0.0

## Guide

- This code is very simple, it should explain itself.
- Train a model from scratch
  - Prepare training corpus and vocab
    - use `preprocess.py`, see more details there
  - Training
    - `sh train.sh`
  - For hyper-parameter and all other settings, see the argument parsers in the above two files. We provide a piece of raw text from zhwiki in `toy` folder.

  - Distributed training settings (in `train.sh`)
    - --world_size #total number of gpus
    - --gpus #gpus on this machine
    - --MASTER_ADDR #master node IP
    - --MASTER_PORT #master node port
    - --start_rank # range from 0 to world_size-1, the index of the first gpu on this machine
    - --backend # 'nccl' or 'gloo', nccl is generally better but may not work on some machines

- Exemplar use of a trained model

  - See [`sentence_pair_matching`](./sentence_pair_matching) for more details.

  - Preprocessing Guide

    ```python
    from google_bert import BasicTokenizer
    tokenizer = BasicTokenizer()
    x = "BERT在多个自然语言处理任务中表现优越。"
    char_level_tokens = tokenizer.tokenize(x)
    
    # if processing at word level
    # We assume a word segmenter "word_segmenter" in hand
    word_level_tokens = word_segmenter.segment(x)
    #Note you may need to add speical tokens (e.g., [CLS], [SEP]) by yourself.
    ```
