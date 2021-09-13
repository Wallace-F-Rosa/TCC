# !/bin/bash

# B bronchiseptica and T retortaeformis coinfectio 
mkdir -p "testes/${1}"
cd "testes/${1}"
python ../../redes_reg_genes/redes_pesos/load_graph.py "redes_reg_genes/redes_bio/40-100/${1}/expr/expressions.ALL.txt" 1
python ../../TCC/generate_code.py expressions_pesos.txt
nvcc -w -DTHREADS=256 -arch=sm_37 -Xcompiler -fopenmp -Wno-deprecated-gpu-targets -lcurand -o tlf tlf.cu
./tlf 1000 > 1K.txt
./tlf 10000 > 10K.txt
./tlf 100000 > 100K.txt
./tlf 1000000 > 1M.txt
./tlf 10000000 > 10M.txt
