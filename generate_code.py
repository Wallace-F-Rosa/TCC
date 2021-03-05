import os
import sys

def generateCudaCode(weights_file_path):
    # lendo pesos das redes
    weightsFile = open(weights_file_path, 'r')
    fileContent = weights_file.readlines()
    weightsFile.close()

    #definindo valores
    networkSize = fileContent[0].split('\n')[0]
    weightsSize = fileContent[1].split('\n')[1].split(' ')
    # gerando código da rede
    code_file = open('tlf.cu', 'w+')
    code_file.write('#include <iostream>\n')
    code_file.write('#include <chrono>\n')
    code_file.write('#include <ctime>\n')
    code_file.write('#include <string>\n')
    code_file.write('#include <limits>\n')
    code_file.write('#include <stdio.h>\n')
    code_file.write('#include <stdlib.h>\n')
    code_file.write('struct { ulonglong['+str(networkSize//64)+'] } estado;')
    code_file.close()

if __name__ == '__main__' :
    try :
        weights_file_path = sys.argv[0]
        if not os.path.exists(weights_file_path) :
            print('Arquivo com pesos da rede não foi encontrado!')
        else:
            generateCudaCode(weights_file_path)
    except :
        print('Arquivo com pesos da rede deve ser fornecido como parâmetro do programa!')
