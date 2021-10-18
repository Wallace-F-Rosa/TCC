# !/bin/bash

# gpu e openmp
echo "Teste Simulações GPU e CPU Openmp"
./test-all.sh
zip -r testes-both-tlf.zip testes
rm -rf testes
./test-all-bool.sh
zip -r testes-both-bool.zip testes
rm -rf testes

# single core
echo "Teste Simulações CPU Single Core"
./test-all-singlecore.sh
zip -r testes-singlecore-tlf.zip testes
rm -rf testes
./test-all-bool-singlecore.sh
zip -r testes-singlecore-bool.zip testes
rm -rf testes
