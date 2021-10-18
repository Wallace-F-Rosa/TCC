# !/bin/bash

echo "Teste redes disponíveis em redes_reg_genes"

echo "Cholesterol Regulatory Pathway"
./test-singlecore.sh "Cholesterol Regulatory Pathway" 1024 -b

echo "Apoptosis network"
./test-singlecore.sh "Apoptosis network"  1024 -b

echo "Guard Cell Abscisic Acid Signaling"
./test-singlecore.sh "Guard Cell Abscisic Acid Signaling" 1024 -b

echo "Differentiation of T lymphocytes"
./test-singlecore.sh "Differentiation of T lymphocytes" 1024 -b

echo "B bronchiseptica and T retortaeformis coinfectio"
./test-singlecore.sh "B bronchiseptica and T retortaeformis coinfectio" 512 -b

echo "MAPK Cancer Cell Fate Network"
./test-singlecore.sh "MAPK Cancer Cell Fate Network" 256 -b

echo "T-LGL Survival Network 2011"
./test-singlecore.sh "T-LGL Survival Network 2011" 512 -b

echo "PC12 Cell Differentiation"
./test-singlecore.sh "PC12 Cell Differentiation" 256 -b

echo "Bortezomib Responses in U266 Human Myelo"
./test-singlecore.sh "Bortezomib Responses in U266 Human Myelo" 1024 -b

echo "HGF Signaling in Keratinocytes"
./test-singlecore.sh "HGF Signaling in Keratinocytes" 1024 -b

echo "Glucose Repression Signaling 2009"
./test-singlecore.sh "Glucose Repression Signaling 2009" 512 -b

echo "Yeast Apoptosis"
./test-singlecore.sh "Yeast Apoptosis" 512 -b

echo "Lymphopoiesis Regulatory Network"
./test-singlecore.sh "Lymphopoiesis Regulatory Network" 256 -b

echo "IL-6 Signalling"
./test-singlecore.sh "IL-6 Signalling" 256 -b

echo "EGFR & ErbB Signaling"
./test100-singlecore.sh "EGFR & ErbB Signaling" 256 -b

echo "signal transduction in fibroblasts"
./test100-singlecore.sh "signal transduction in fibroblasts" 256 -b
