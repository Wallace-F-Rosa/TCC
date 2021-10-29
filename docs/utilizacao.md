# Utilização

Clonar repositório :

```
git clone https://github.com/Wallace-F-Rosa/TCC.git dir_name
cd dir_name
git submodule update --init --recursive
```

Instalação de dependências:

`pip install -r requirements.txt`

### Geração de código
Abaixo se encontram as instruções para gerar Código CUDA utilizando o script `generate_code.py`.
Caso deseje gerar código C++ + Openmp para utilzar somente a cpu, utilize a flag `--cpu`.

#### Equações de soma de pesos:
Para gerar as equações de soma de pesos utilizando equações booleanas utilize :

`python redes_reg_genes/redes_pesos/load_graph.py rede.txt 1`

Para gerar o código necessário utilize o comando abaixo utilizando o caminho até o arquivo .txt com os pesos da rede.

`python TCC/generate_code.py caminho/ate/expressoes/rede_pesos.txt`

#### Equações booleanas:
Para gerar código utilizando utilize o comando abaixo utilizando o caminho até o arquivo .txt com as 
equações booleanas da rede.

`python TCC/generate_code.py caminho/ate/expressoes/rede.txt -b`

Ambos passos acima devem gerar um arquivo de saída `tlf.cu` com código CUDA para simulações utilizando GPU e `tlf.cpp`
caso seja código que usará somente a CPU.

### Cálculo de atratores

#### Compilar código para GPU
Para compilar o código CUDA utilize o `nvcc` com algumas opções que garantem o uso da biblioteca [cuRAND](https://docs.nvidia.com/cuda/curand/introduction.html#introduction) para geração de números aleatórios utilizando GPU.

`nvcc -w -Xcompiler -Wno-deprecated-gpu-targets -lcurand -o tlf tlf.cu`

#### Compilar código para CPU
A compilação CPU depende do compilador `g++` também com flags para garantir o uso da biblioteca [Openmp](https://www.openmp.org/) 

`g++ -std=c++11 -fopenmp -o tlf tlf.cpp`

#### Executar simulação
Para executar basta executar o programa `tlf` informando a quantidade de estados a serem utilizadas na simulação da rede.
Este programa gera um arquivo `atractors.json` para ser utilizado na visualização.

Exemplo de uso:

`./tlf 1000`

### Visualização de atratores
A visualização utiliza a ferramente [NetworkX](https://networkx.org/) para definição dos grafos dos atratores e a ferramenta [pyvis](https://pyvis.readthedocs.io/en/latest/) para plotagem da visualização.

Para visualizar os atratores basta executar o script `viz-atractors.py` passando como parâmetro o caminho até o arquivo `atractors.json` resultado dos passos anteriores.

`python vis-atractors.py atractors.json`

## TroubleShooting

### CUDA Error "too many resources requested for launch"

Este erro ocorre quando o número de registradores necessário para executar o código CUDA
excede o disponível na GPU. Para contonar o erro é necessário utilizar a flag de compilação 
THREADS informando manualmente o número de threads a serem utilizadas utilizando um número 
menor que o máximo disponível(disponível ao executar a simulação). Uma estratégia válida é 
utilizar metade do máximo permitido, e, se o erro persistir utilizar utilizar a metade deste e assim por diante.

Exemplo:

`nvcc -w -DTHREADS=256 -Xcompiler -Wno-deprecated-gpu-targets -lcurand -o tlf tlf.cu`
