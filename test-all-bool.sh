# !/bin/bash

echo "Teste redes disponíveis em redes_reg_genes"

echo "Cholesterol Regulatory Pathway"
./test.sh "Cholesterol Regulatory Pathway" 1024 -b

echo "Apoptosis network"
./test.sh "Apoptosis network"  1024 -b

echo "Guard Cell Abscisic Acid Signaling"
./test.sh "Guard Cell Abscisic Acid Signaling" 1024 -b

echo "Differentiation of T lymphocytes"
./test.sh "Differentiation of T lymphocytes" 1024 -b

echo "B bronchiseptica and T retortaeformis coinfectio"
./test.sh "B bronchiseptica and T retortaeformis coinfectio" 512 -b

echo "MAPK Cancer Cell Fate Network"
./test.sh "MAPK Cancer Cell Fate Network" 256 -b

echo "T-LGL Survival Network 2011"
./test.sh "T-LGL Survival Network 2011" 512 -b

echo "PC12 Cell Differentiation"
./test.sh "PC12 Cell Differentiation" 256 -b

echo "Bortezomib Responses in U266 Human Myelo"
./test.sh "Bortezomib Responses in U266 Human Myelo" 1024 -b

echo "HGF Signaling in Keratinocytes"
./test.sh "HGF Signaling in Keratinocytes" 1024 -b

echo "Glucose Repression Signaling 2009"
./test.sh "Glucose Repression Signaling 2009" 512 -b

echo "Yeast Apoptosis"
./test.sh "Yeast Apoptosis" 512 -b

echo "Lymphopoiesis Regulatory Network"
./test.sh "Lymphopoiesis Regulatory Network" 256 -b

echo "IL-6 Signalling"
./test.sh "IL-6 Signalling" 256 -b

echo "EGFR & ErbB Signaling"
./test100.sh "EGFR & ErbB Signaling" 256 -b

echo "signal transduction in fibroblasts"
./test100.sh "signal transduction in fibroblasts" 256 -b
