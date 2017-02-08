The ori_data file contain bigram phrase similarity datasets with all human annotation in both Chinese and English. We also include the instruction of how we construct the datasets.

The code file include train/dev/test dataset and all embeddings and codes in paper 'Comparison Study on Critical Components in Composition Model for Phrase Representation'. The code is written in python and requires numpy, scipy, theano and the lasagne library. The code is modified from https://github.com/jwieting/iclr2016

If you use our code for your work please cite:

@article{wang2017comparison,
  title={Comparison Study on Critical Components in Composition Model for Phrase Representation},
  author={Wang, Shaonan and Zong, Chengqing},
  journal={ACM Transactions on Asian and Low-Resource Language Information Processing (TALLIP)},
  volume={16},
  number={3},
  pages={16},
  year={2017},
  publisher={ACM}
}