# !/bin/bash

# B bronchiseptica and T retortaeformis coinfectio 
mkdir -p "testes/${1}"
cd "testes/${1}"
python ../../redes_reg_genes/redes_pesos/load_graph.py "../../redes_reg_genes/redes_bio/40-100/${1}/expr/expressions.ALL.txt" 1
if [[ $# -ge 3 ]]
then
    python ../../generate_code.py "../../redes_reg_genes/redes_bio/40-100/${1}/expr/expressions.ALL.txt" $3 --cpu --single-core
else
    python ../../generate_code.py "expressions_pesos.txt" --cpu --single-core
fi
g++ -fopenmp -o tlf tlf.cpp
./tlf 1000 > 1K.txt
./tlf 10000 > 10K.txt
./tlf 100000 > 100K.txt
./tlf 1000000 > 1M.txt
./tlf 10000000 > 10M.txt
