# -*- coding: utf-8 -*-

import click
import wget

from abstract_db import AbstractDB


def download_dbpedia_abstract_files(out_dir):
    for n in range(114):
        url = 'http://downloads.dbpedia.org/2015-04/ext/nlp/abstracts/en/abstracts_en%d.ttl.gz' % (n,)
        click.echo('Getting %s' % url)
        # subprocess.getstatusoutput('wget -P %s/ %s' % (out_dir, url))  # unix
        # subprocess.getstatusoutput('curl -o %s %s' % (out_dir, url))  # window
        filename = wget.detect_filename(url)
        wget.download(url)



def build_abstract_db(in_dir, out_file, pool_size):
    AbstractDB.build(in_dir, out_file, pool_size)




if __name__ == "__main__":
    # 第一步：下载DBpedia abstract corpus语料
    # download_dbpedia_abstract_files(out_dir='./dbpedia_abstract_corpus')
    # 第二步：
    build_abstract_db(in_dir='./dbpedia_abstract_corpus', out_file="./dbpedia_abstract.db", pool_size=10)

