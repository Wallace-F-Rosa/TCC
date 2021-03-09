import os
import sys

def generateCudaCode(weights_file_path):
    """ Gera código cuda para simulação da rede utilizando equações com peso """

    # lendo pesos das redes
    weightsFile = open(weights_file_path, 'r')
    fileContent = weightsFile.readlines()
    weightsFile.close()

    # definindo valores
    networkSize = int(fileContent[0].split('\n')[0])
    weightsSize = [ int(x) for x in fileContent[1].split('\n')[0].split(' ')]
    print(weightsSize)
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

    #função de comparação entre estados
    code_file.write('__device__ bool equals(state a, state b) {\n')
    code_file.write('   for (int i = 0; i < '+str(stateSize)+'; i++) {\n')
    code_file.write('       if (a[i] != b[i])\n')
    code_file.write('           return false;\n')
    code_file.write('   }\n')
    code_file.write('   return true;\n')
    code_file.write('}\n')

    # cuda kernel recebe os estados aleatórios inicialmente, simulando N estados até o número de simulações fornecido
    code_file.write('__global__ void network_simulation(state * randState, state * statef, unsigned long long SIMULATIONS) {\n')
    code_file.write('   unsigned long long tid = threadIdx.x + blockIdx.x*blockDim.x;\n')
    code_file.write('   state state0, state1;\n')
    code_file.write('   if (tid < SIMULATIONS) {\n')
    
    # inicializando estados
    for i in range(stateSize):
        code_file.write('       state0['+str(i)+'] = randState['+str(i)+'];\n')
        code_file.write('       state1['+str(i)+'] = 0;\n')

    code_file.write('       do {\n')

    # equações : 
    # estadof[var//64] = 0
    # estadof[var//64] |= ( ( ( (estado0[i//64] >> var) % 2 )*peso + ( (estado0[i//64] >> var) % 2 )*peso  ...) >= lim) << var;
    # gerando equações do passo 1 (estado0 anda um passo)
    # FIXME: kernel não roda quando temos muitas instruções de equação
    for i in range(networkSize) :
        eq = '          state1['+str(i//64)+'] |= ( ( '
        line = fileContent[2+i].split('\n')[0].split(' ')
        for y in range(weightsSize[i]):
            eq += '( ( state0['+str(i//64)+'] >> '+str(line[2*y])+') % 2 ) * '+str(line[2*y+1])
        eq += ' ) >= '+str(line[len(line)-1])+' ) << '+str(i)+';\n'
        code_file.write(eq)

    # estado0 recebe estado1, andamos 1 passo com as equações da rede
    for i in range(stateSize):
        code_file.write('       state0['+str(i)+'] = state1['+str(i)+'];\n')

    # aplicamos as equações novamente em estado1 para andar 2 passos
    for i in range(networkSize) :
        eq = '          state1['+str(i//64)+'] |= ( ( '
        line = fileContent[2+i].split('\n')[0].split(' ')
        for y in range(weightsSize[i]):
            eq += '( ( state0['+str(i//64)+'] >> '+str(line[2*y])+') % 2 ) * '+str(line[2*y+1])
        eq += ' ) >= '+str(line[len(line)-1])+' ) << '+str(i)+';\n'
        code_file.write(eq)
            
    code_file.write('       } while(equals(state0, state1));\n')

    # salva o estado inicial do atrator na memória global da gpu
    for i in range(stateSize) :
        code_file.write('       statef[tid]['+str(i)+'] = state1['+str(i)+'];\n')
    code_file.write('   }\n')
    code_file.write('}\n')

    code_file.write('\n')
    code_file.write('void init_rand(state * randState, unsigned long long SIMULATIONS) {\n')
    code_file.write('   srand(time(NULL))\n')
    code_file.write('   for (unsigned long long i = 0; i < SIMULATIONS; i++) {\n')
    for i in range(stateSize):
        code.write('        randState[i]['+str(i)+'] = rand()%((unsigned long)(1<<31)-1);\n')
    code_file.write('       \n')
    code_file.write('   }\n')
    code_file.write('   \n')
    code_file.write('}\n')

    # TODO: código main, alocar vetores e preencher estados iniciais com números randômicos
    code_file.write('int main(int argc) {\n')
    code_file.write('   \n')
    code_file.write('}\n')

    code_file.close()

if __name__ == '__main__' :
    """ Função main recebe arquivo de rede como parâmetro e gera como saída
    um arquivo tlf.cu para simulação da rede em CUDA"""
    # try :
    weights_file_path = sys.argv[1]
    print(weights_file_path)
    if not os.path.exists(weights_file_path) :
        print('Arquivo com pesos da rede não foi encontrado!')
    else:
        generateCudaCode(weights_file_path)
# except Exception as e:
    #     print(e)
