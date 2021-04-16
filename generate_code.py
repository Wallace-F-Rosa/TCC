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
    code_file.write('#include <iostream>\n'+
                    '#include <chrono>\n'+
                    '#include <ctime>\n'+
                    '#include <string>\n'+
                    '#include <sstream>\n'+
                    '#include <vector>\n'+
                    '#include <map>\n'+
                    '#include <unordered_map>\n'+
                    '#include <limits>\n'+
                    '#include <stdio.h>\n'+
                    '#include <stdlib.h>\n'+
                    '#include <algorithm>\n'+
                    '\nusing namespace std;\n')

    # estado é um vetor de inteiros
    # cada bit representa um vértice
    stateSize = networkSize//64 + (networkSize%64 != 0)
    code_file.write('typedef unsigned long long state['+str(stateSize)+'];\n')

    # função de comparação entre estados kernel
    code_file.write('__device__ bool equals_d(state a, state b) {\n'+
                    '   for (int i = 0; i < '+str(stateSize)+'; i++) {\n'+
                    '       if (a[i] != b[i])\n'+
                    '           return false;\n'+
                    '   }\n'+
                    '   return true;\n'+
                    '}\n')

    # função de comparação entre estados host
    code_file.write('bool equals_h(state a, state b) {\n'+
                    '   for (int i = 0; i < '+str(stateSize)+'; i++) {\n'+
                    '       if (a[i] != b[i])\n'+
                    '           return false;\n'+
                    '   }\n'+
                    '   return true;\n'+
                    '}\n')

    # cuda kernel recebe os estados aleatórios inicialmente, simulando N estados até o número de simulações fornecido
    code_file.write('__global__ void network_simulation_d(state * statef, unsigned long long SIMULATIONS) {\n'+
                    '   unsigned long long tid = threadIdx.x + blockIdx.x*blockDim.x;\n'+
                    '   state state0, state1, aux;\n'+
                    '   if (tid < SIMULATIONS) {\n')
    
    # inicializando estados
    for i in range(stateSize):
        code_file.write('       state0['+str(i)+'] = state1['+str(i)+'] = statef[tid]['+str(i)+'];\n'+
                        '       aux['+str(i)+'] = 0;\n')

    code_file.write('       do {\n')

    # GPU
    # equações : 
    # estadof[var//64] = 0
    # estadof[var//64] |= ( ( ( (estado0[i//64] >> var) % 2 )*peso + ( (estado0[i//64] >> var) % 2 )*peso  ...) >= lim) << var;
    # gerando equações do passo 1 (estado0 anda um passo)
    # FIXME: kernel não roda quando temos muitas instruções de equação
    for i in range(networkSize) :
        eq = '          aux['+str(i//64)+'] |= (unsigned long long) ( ( '
        line = fileContent[2+i].split('\n')[0].split(' ')
        for y in range(weightsSize[i]):
            eq += '( ( state0['+str(i//64)+'] >> '+str(line[2*y])+') % 2 ) * '+str(line[2*y+1])
            if y != weightsSize[i] - 1:
                eq+=' + '
        eq += ' ) >= '+str(line[len(line)-1])+' ) << '+str(i%64)+';\n'
        code_file.write(eq)

    # estado0 recebe estado1, andamos 1 passo com as equações da rede
    for i in range(stateSize):
        code_file.write('       state0['+str(i)+'] = state1['+str(i)+'];\n'+
                        '       aux['+str(i)+'] = 0;\n')

    # aplicamos as equações novamente em estado1 para andar 2 passos
    for i in range(networkSize) :
        eq = '          aux['+str(i//64)+'] |= (unsigned long long) ( ( '
        line = fileContent[2+i].split('\n')[0].split(' ')
        for y in range(weightsSize[i]):
            eq += '( ( state0['+str(i//64)+'] >> '+str(line[2*y])+') % 2 ) * '+str(line[2*y+1])
            if y != weightsSize[i] - 1:
                eq+=' + '
        eq += ' ) >= '+str(line[len(line)-1])+'ULL ) << '+str(i%64)+';\n'
        code_file.write(eq)
    
    # estado1 recebe aux, andamos 2 passos com as equações da rede
    for i in range(stateSize):
        code_file.write('           state1['+str(i)+'] = aux['+str(i)+'];\n'+
                        '           aux['+str(i)+'] = 0;\n')

    code_file.write('       } while(!equals_d(state0, state1));\n')

    # salva o estado inicial do atrator na memória global da gpu
    for i in range(stateSize) :
        code_file.write('       statef[tid]['+str(i)+'] = state1['+str(i)+'];\n')
    code_file.write('   }\n'+
                    '}\n')

    code_file.write('\n')

    # CPU
    # versão cpu do calculo de atratores
    code_file.write('void network_simulation_h(state * statef, unsigned long long SIMULATIONS){\n'+
                    '   for(unsigned long long i = 0; i < SIMULATIONS; i++){\n'+
                    '       state state0, state1, aux;\n')
    # inicializando estados
    for i in range(stateSize):
        code_file.write('       state0['+str(i)+'] = state1['+str(i)+'] = statef[i]['+str(i)+'];\n'+
                        '       aux['+str(i)+'] = 0;\n')
    code_file.write('       do {\n')
    for i in range(networkSize) :
        eq = '          aux['+str(i//64)+'] |= (unsigned long long) ( ( '
        line = fileContent[2+i].split('\n')[0].split(' ')
        for y in range(weightsSize[i]):
            eq += '( ( state0['+str(i//64)+'] >> '+str(line[2*y])+') % 2 ) * '+str(line[2*y+1])
            if y != weightsSize[i] - 1:
                eq+=' + '
        eq += ' ) >= '+str(line[len(line)-1])+' ) << '+str(i%64)+';\n'
        code_file.write(eq)

    # estado0 recebe estado1, andamos 1 passo com as equações da rede
    for i in range(stateSize):
        code_file.write('           state0['+str(i)+'] = state1['+str(i)+'] = aux['+str(i)+'];\n'+
                        '           aux['+str(i)+'] = 0;\n')

    # aplicamos as equações novamente em estado1 para andar 2 passos
    for i in range(networkSize) :
        eq = '          aux['+str(i//64)+'] |= (unsigned long long) ( ( '
        line = fileContent[2+i].split('\n')[0].split(' ')
        for y in range(weightsSize[i]):
            eq += '( ( state0['+str(i//64)+'] >> '+str(line[2*y])+') % 2 ) * '+str(line[2*y+1])
            if y != weightsSize[i] - 1:
                eq+=' + '
        eq += ' ) >= '+str(line[len(line)-1])+'UL ) << '+str(i%64)+';\n'
        code_file.write(eq)

    # estado1 recebe aux, andamos 2 passos com as equações da rede
    for i in range(stateSize):
        code_file.write('           state1['+str(i)+'] = aux['+str(i)+'];\n'+
                        '           aux['+str(i)+'] = 0;\n')
            
    code_file.write('       } while(!equals_h(state0, state1));\n')

    # salva o estado inicial do atrator na memória global da gpu
    for i in range(stateSize) :
        code_file.write('       statef[i]['+str(i)+'] = state1['+str(i)+'];\n')
    code_file.write('   }\n'+
                    '}\n')

    # função que converte estado para string
    code_file.write("string to_string(state s){\n"+
                    "   string result;\n"+
                    "   stringstream stream;\n"+
                    "   stream << s[0];"
                    "   for(int i = 1; i < "+str(stateSize-1)+"; i++)\n"+
                    "       stream << '|' << s[i];\n"+
                    "   stream >> result;\n"+
                    "   return result;\n"+
                    "}\n")

    # função que converte um atrator(vetor de estados) para string
    code_file.write("string to_string(vector<string> atractor){\n"+
                    # gambiarra para imprimir aspas duplas
                    '   if(atractor.size() == 0) return "'+'";\n'+
                    "   string result = atractor[0];\n"+
                    "   for (int i = 1; i < atractor.size(); i++)\n"+
                    # gambiarra para imprimir aspas duplas
                    '       result += "' + ' " + atractor[i];\n'+
                    "   return result;\n"+
                    "}\n")

    # função que recebe um estado de um atrator e entrega o atrator completo
    # em um vector<string> com representação em string do atrator
    code_file.write("vector<string> getAtractor(state s) {\n"+
                    "   state s0,s1,aux;\n"+
                    '   vector<string> atractor; atractor.push_back(to_string(s0));\n'+
                    "   for (int i = 0; i < "+str(stateSize)+"; i++){\n"+
                    "       s0[i] = s1[i] = s[i];\n"+
                    "       aux[i] = 0;\n"+
                    "   }\n"+
                    "   while(true) {\n")

    for i in range(networkSize) :
        eq = '          aux['+str(i//64)+'] |= (unsigned long long) ( ( '
        line = fileContent[2+i].split('\n')[0].split(' ')
        for y in range(weightsSize[i]):
            eq += '( ( s1['+str(i//64)+'] >> '+str(line[2*y])+') % 2 ) * '+str(line[2*y+1])
            if y != weightsSize[i] - 1:
                eq+=' + '
        eq += ' ) >= '+str(line[len(line)-1])+'ULL ) << '+str(i%64)+';\n'
        code_file.write(eq)
 
    for i in range(stateSize):
        code_file.write('           s1['+str(i)+'] = aux['+str(i)+'];\n'+
                        '           aux['+str(i)+'] = 0;\n')

    code_file.write("       if (!equals_h(s0,s1))\n"+
                    "           atractor.push_back(to_string(s1));\n"+
                    "       else\n"+
                    "           break;\n"+
                    "   }\n"+
                    "   sort(atractor.begin(), atractor.end());\n"+
                    "   return atractor;\n"+
                    "}\n")

    # função que junta os atratores a partir dos estados encontrados na simulação
    # atrator é um string com os estados
    # FIXME: retorna atatores vazions
    code_file.write('vector<string> complete_atractors(state * st, unsigned long long SIMULATIONS){\n'+
                    '   vector<string> atractors;\n'
                    '   map<string, string> state_to_at;\n'+
                    '   unordered_map<string, unsigned long> at_freq;\n'+
                    '   for(unsigned long long i = 0; i < SIMULATIONS; i++){\n'+
                    '       if (state_to_at.count(to_string(st[i])) > 0) {\n'+
                    '           at_freq[state_to_at[to_string(st[i])]]++;\n'+
                    '       } else {\n'+
                    '           vector<string> at = getAtractor(st[i]);\n'+
                    '           string sat = to_string(at);\n'+
                    '           atractors.push_back(sat);\n'+
                    '           for (int i = 0; i < at.size(); i++)\n'+
                    '               state_to_at[at[i]] = sat;\n'+
                    '           at_freq[sat]=1;\n'+
                    '       }\n'+
                    '   }\n'+
                    '   return atractors;\n'+
                    '}\n')

    # função que imprime atratores encontrados num arquivo
    code_file.write('void output_atractors(const vector<string> &atractors) {\n'+
                    '   for (unsigned long long i = 0; i < atractors.size(); i++) {\n'+
                    "       cout << atractors[i] <<" + repr('\n') + ";\n"+
                    '   }\n'+
                    '}\n')

    # inicializa estados inicias aleatoriamente
    code_file.write('void init_rand(state * randState, unsigned long long SIMULATIONS) {\n')
    code_file.write('   srand(time(NULL));\n')
    code_file.write('   for (unsigned long long i = 0; i < SIMULATIONS; i++) {\n')
    for i in range(stateSize):
        code_file.write('        randState[i]['+str(i)+'] = rand()%((unsigned long long)(1ULL<<63)-1ULL);\n')
    code_file.write('       \n')
    code_file.write('   }\n')
    code_file.write('   \n')
    code_file.write('}\n')

    # código main, aloca vetores e preencher estados iniciais com números randômicos
    # chama kernel gpu e função cpu para calcular os atratores
    # TODO: comparar saida dos atratores para ver qual a diferença
    code_file.write('int main(int argc, char **argv) {\n'+
                    '   unsigned long long SIMULATIONS = 0;\n'+    
                    '   std::string argv2 = argv[1];\n'+
                    '   for(int i = 0; i < argv2.size() ; i++)\n'+
                    "       SIMULATIONS += ((unsigned long int)(argv2[i] - '0'))*pow(10,argv2.size()-i-1);\n"+
                    '   state * state_cpu, * statef_h, * statef_d;\n'+
                    '   cout << "Alocating memory...";\n'+
                    '   state_cpu = new state[SIMULATIONS];\n'+
                    '   statef_h = new state[SIMULATIONS];\n'+
                    '   cudaMalloc((state **)&statef_d,sizeof(state)*SIMULATIONS);\n'+
                    '   init_rand(state_cpu, SIMULATIONS);\n'+
                    '   cudaMemcpy(statef_d, state_cpu, sizeof(state)*SIMULATIONS, cudaMemcpyHostToDevice);\n'+
                    '   int threads = 1024;\n'+
                    '   dim3 block(threads);\n'+
                    '   dim3 grid((SIMULATIONS + block.x -1)/block.x);\n'+
                    '   cout << "[OK]" << '+repr("\n")+';\n'+
                    '   cout << "Running Simulation...";\n'+
                    '   network_simulation_d<<<grid,block>>>(statef_d, SIMULATIONS);\n'+
                    '   network_simulation_h(statef_h, SIMULATIONS);\n'
                    '   cout << "[OK]" << '+repr("\n")+';\n'+
                    '   cudaDeviceSynchronize();\n'+
                    '   cout << "Getting atractors found...";\n'+
                    '   network_simulation_d<<<grid,block>>>(statef_d, SIMULATIONS);\n'+
                    '   cudaMemcpy(statef_h, statef_d, sizeof(state)*SIMULATIONS, cudaMemcpyDeviceToHost);\n'+
                    '   vector<string> atratores = complete_atractors(statef_h, SIMULATIONS);\n'
                    '   cout << "[OK]" << '+repr("\n")+';\n'+
                    '   output_atractors(atratores);\n'
                    '   return 0;\n'+
                    '}\n')

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
