# the entity vocab is too large, i.e., about 11M, we need to reduce it
# 1. remove too long entity
# 2. remove with non-english char
# 3. remove count is less than N


def read_vocab_count(file):
    vocab_count_dict = {}
    with open(file, encoding='utf-8') as f:
        next(f)
        count = 0
        for line in f:
            # print(line)
            count += 1
            # if count > 10000:
            #     break
            tmp = line.strip().split('\t')
            if len(tmp) != 2:
                continue
            entity, times = tmp
            vocab_count_dict[entity] = int(times)

    new_vocab_count_dict = {}
    for entity, entity_count in vocab_count_dict.items():
        # if 10 <= entity_count:
        e = entity.split('_')
        if len(e) >= 6:
            continue
        else:
            new_vocab_count_dict[entity] = entity_count

    return new_vocab_count_dict


def reduce_vocab(file):
    vocab_count_dic = read_vocab_count(file)
    from collections import defaultdict
    import collections
    count_count_dict = defaultdict(int)
    for en, count in vocab_count_dic.items():
        count_count_dict[count] += 1

    count_count_dict = collections.OrderedDict(sorted(count_count_dict.items()))
    reverse_count_count_dict = collections.OrderedDict(sorted(count_count_dict.items(), reverse=True))

    import matplotlib.pyplot as plt
    import numpy as np

    plt.style.use('mpl20')

    # make the data
    np.random.seed(3)
    x = list(count_count_dict.keys())
    y = list(count_count_dict.values())
    # size and color:
    sizes = np.random.uniform(15, 80, len(x))
    colors = np.random.uniform(15, 80, len(x))

    # plot
    fig, ax = plt.subplots()

    ax.scatter(x, y, s=sizes, c=colors, vmin=0, vmax=100)

    # ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
    #        ylim=(0, 8), yticks=np.arange(1, 8))

    plt.show()

if __name__ == "__main__":
    reduce_vocab(file="G:\D\MSRA\knowledge_aware\Annotated-WikiExtractor-master/annotated_wikiextractor/entity_vocab_count.tsv")