# KFormers
<img src="resources/kformers_logo.png" width="200" alt="KFormers">

[![CircleCI](https://circleci.com/gh/studio-ousia/luke.svg?style=svg&circle-token=49524bfde04659b8b54509f7e0f06ec3cf38f15e)](https://circleci.com/gh/studio-ousia/luke)

---

## Resources
1. dbpedia_abstract_corpus
2. LUKE checkpoint
3. Wikimapper

## Released Models

## Reproducing Experimental Results

### Entity Typing on Open Entity Dataset

### Relation Classification on TACRED Dataset

**Fine-tuning the model:**

## Citation

## Contact Info

# BUG
> 1. 预训练的时候，当torch.load加载的数据量小时，代码没有问题；当数据量大时，例如34M，128长度，代码报如下错误。且check代码发现
> `torch.barrier`之间的code在主进程执行完后，其他进程也会执行，且在其他进程执行完后，报如下错误。
`Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed.`
> 
> 修改方法，将代码中的   `print`换成`logging.info`即可
