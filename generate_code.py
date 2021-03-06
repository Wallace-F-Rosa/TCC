import os
import sys

def generateCudaCode(weights_file_path):
    # lendo pesos das redes
    weightsFile = open(weights_file_path, 'r')
    fileContent = weightsFile.readlines()
    weightsFile.close()

    # definindo valores
    networkSize = int(fileContent[0].split('\n')[0])
    weightsSize = fileContent[1].split('\n')[0].split(' ')
    # gerando código da rede
    code_file = open('tlf.cu', 'w+')
    
    # headers do código
    code_file.write('#include <iostream>\n')
    code_file.write('#include <chrono>\n')
    code_file.write('#include <ctime>\n')
    code_file.write('#include <string>\n')
    code_file.write('#include <limits>\n')
    code_file.write('#include <stdio.h>\n')
    code_file.write('#include <stdlib.h>\n')

    # estado é um vetor de inteiros
    # cada bit representa um vértice
    stateSize = networkSize//64 + (networkSize%64 != 0)
    code_file.write('typedef ulonglong['+str(stateSize)+'] state;\n')

    # cuda kernel recebe os estados aleatórios inicialmente, simulando N estados até o número de simulações fornecido
    code_file.write('__global__ void network_simulation(state * randState, state * statef, unsigned long long SIMULATIONS) {\n')
    code_file.write('   unsigned long long tid = threadIdx.x + blockIdx.x*blockDim.x;\n')
    code_file.write('   state state0, state1;\n')
    code_file.write('   if (tid < SIMULATIONS) {\n')
    
    # inicializando estados
    for i in range(stateSize):
        code_file.write('       state0['+str(i)+'] = randState['+str(i)+'];\n')

    # TODO : gerar equações da rede
    for i in range(networkSize) :
        eq = '      state1['+str(i//64)+'] |= '
        line = fileContent[2+i].split('\n')[0].split(' ')
        for y in range(len(weightsSize)):
            eq += ''
            

    # salva o estado inicial do atrator na memória global da gpu
    for i in range(stateSize) :
        code_file.write('       statef[tid]['+str(i)+'] = state1['+str(i)+'];\n')
    code_file.write('   }\n')
    code_file.write('}\n')
    code_file.close()

if __name__ == '__main__' :
    try :
        weights_file_path = sys.argv[1]
        print(weights_file_path)
        if not os.path.exists(weights_file_path) :
            print('Arquivo com pesos da rede não foi encontrado!')
        else:
            generateCudaCode(weights_file_path)
    except Exception as e:
        print(e)
