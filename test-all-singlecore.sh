# !/bin/bash

echo "Teste redes disponíveis em redes_reg_genes"

echo "Cholesterol Regulatory Pathway"
./test-singlecore.sh "Cholesterol Regulatory Pathway" 1024

echo "Apoptosis network"
./test-singlecore.sh "Apoptosis network"  1024

echo "Guard Cell Abscisic Acid Signaling"
./test-singlecore.sh "Guard Cell Abscisic Acid Signaling" 1024

echo "Differentiation of T lymphocytes"
./test-singlecore.sh "Differentiation of T lymphocytes" 1024

echo "B bronchiseptica and T retortaeformis coinfectio"
./test-singlecore.sh "B bronchiseptica and T retortaeformis coinfectio" 512

echo "MAPK Cancer Cell Fate Network"
./test-singlecore.sh "MAPK Cancer Cell Fate Network" 256

echo "T-LGL Survival Network 2011"
./test-singlecore.sh "T-LGL Survival Network 2011" 512

echo "PC12 Cell Differentiation"
./test-singlecore.sh "PC12 Cell Differentiation" 256

echo "Bortezomib Responses in U266 Human Myelo"
./test-singlecore.sh "Bortezomib Responses in U266 Human Myelo" 1024

echo "HGF Signaling in Keratinocytes"
./test-singlecore.sh "HGF Signaling in Keratinocytes" 1024

echo "Glucose Repression Signaling 2009"
./test-singlecore.sh "Glucose Repression Signaling 2009" 512

echo "Yeast Apoptosis"
./test-singlecore.sh "Yeast Apoptosis" 512

echo "Lymphopoiesis Regulatory Network"
./test-singlecore.sh "Lymphopoiesis Regulatory Network" 256

echo "IL-6 Signalling"
./test-singlecore.sh "IL-6 Signalling" 256

echo "EGFR & ErbB Signaling"
./test100-singlecore.sh "EGFR & ErbB Signaling" 256

echo "signal transduction in fibroblasts"
./test100-singlecore.sh "signal transduction in fibroblasts" 256
