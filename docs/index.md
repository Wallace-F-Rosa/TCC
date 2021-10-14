## Introdução


### Redes Reguladoras

As redes utilizadas como teste, suas equações e referências podem ser encontradas no repositório [redes_reg_genes](https://github.com/Wallace-F-Rosa/redes_reg_genes)(submódulo deste repositório).

## Dependências
Para geração de código CUDA e visualização de atratores é utilizada a linguagem Python com os seguintes módulos:

* pandas
* numpy
* py2cytoscape
* pyvis
* networkx
* dimcli

Para compilação e execução das simulação é necessário ter um ambiente com alguma GPU NVIDIA disponível e o [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) instalado.

## Colab

É possível executar as simulações de rede e visualização de atratores via [Google Colab](https://colab.research.google.com/drive/19riKQhOeUC39oioSliIDZQNYto06Gp21?usp=sharing), sem necessidade de possuir uma GPU NVIDIA localmente.

# Utilização / TroubleShooting

Caso você tenha acesso a um ambiente com GPU NVIDIA (seja localmente ou em ambientes como a AWS) as instruções para gerar o código CUDA necessário e executar as simulações se encontram [aqui](utilizacao.md).
