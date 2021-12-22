# Cross-modal Retrieval using Transformer Encoder Reasoning Networks


## Architecture

## Losses

## Metrics
https://amitness.com/2020/08/information-retrieval-evaluation/

## Datasets


## Execution

- Installation

```
pip install -r requirements.txt
apt install libomp-dev
pip install faiss-gpu
```

- For training
```
%cd main
!PYTHONPATH=. python tools/train.py --config="/content/main/tools/configs/yaml/sgraf.yaml" \
                 --print_per_iter=50 \
                 --val_interval=3 \
                 --save_interval=300 \
                 --saved_path="/content/drive/MyDrive/AI/Weights/flickr30k-retrieval/SGRAF" \
```

- For evaluation
```
%cd /content/main
!python eval.py --config="/content/main/configs/yaml/tern.yaml" \
                --top_k=10 \
                --weight="/content/drive/MyDrive/AI/Weights/flickr30k-retrieval/TERN/TERN_best.pth" \
```



## Notebooks
- [![Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1z38DL7DxjXC-wH_AhC2NIipubgajOkni?usp=sharing) Use FasterRCNN to extract Bottom Up embeddings 
- [![Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/10dRDQkuZ3KZQ_4bwMbevZoiWzP_OVbVJ?usp=sharing) Use BERT to extract text embeddings 

## Results




## Paper References

```
@misc{messina2021transformer,
      title={Transformer Reasoning Network for Image-Text Matching and Retrieval}, 
      author={Nicola Messina and Fabrizio Falchi and Andrea Esuli and Giuseppe Amato},
      year={2021},
      eprint={2004.09144},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

```
@article{JDH17,
  title={Billion-scale similarity search with GPUs},
  author={Johnson, Jeff and Douze, Matthijs and J{\'e}gou, Herv{\'e}},
  journal={arXiv preprint arXiv:1702.08734},
  year={2017}
}
```

## Code References

- https://github.com/facebookresearch/faiss
- https://github.com/KevinMusgrave/pytorch-metric-learning
