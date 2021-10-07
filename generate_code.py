import os
import argparse
import sys

def write_random_initator(code_file, state_size, cpu=False):
    """Escreve no arquivo de saída a função de inicialização dos estados.

    Args:
        code_file (file): objeto representando arquivo de saída aberto para escrita.
        state_size (int): tamanho do estado da rede em representação de números de 64 bits.
        cpu (bool): código gerado deve ser C++ puro sem funcionalidades CUDA.
    """
    if not cpu:
        # gerador de long long sobol64 gpu. chamada no host
        code_file.write('void init_rand_d(unsigned long long * state_d, unsigned long long SIMULATIONS) {\n'+
                        '   curandGenerator_t gen;\n'+
                        '   curandCreateGenerator(&gen, CURAND_RNG_QUASI_SOBOL64);\n'+
                        '   curandGenerateLongLong(gen, state_d, '+ str(state_size) +'*SIMULATIONS);\n'+
                        '   curandDestroyGenerator(gen);\n'+
                        '}\n')

    # números aleatórios cpu
    code_file.write('void init_rand_h(unsigned long long * state, unsigned long long SIMULATIONS) {\n'+
                    '   std::random_device rd;\n'+
                    '   std::mt19937_64 e2(rd());\n'+
                    '   std::uniform_int_distribution<unsigned long long> dist(0, (unsigned long long)std::llround(std::pow(2,64)));\n'+
                    '   for (unsigned long long i = 0; i < SIMULATIONS; i++) {\n'+
                    '       for (size_t j = 0; j < '+ str(state_size) +'; j++)\n'+
                    '           state[i*'+ str(state_size) +' + j] = dist(e2);\n'+
                    '   }\n'+
                    '}\n')

def write_equations(code_file, state_size, network_nodes, eqs_size, eqs_file_content, boolean_equations=False):
    """Escreve no arquivo de saída os comandos para executar as equações diretamente no código.
        
    Args:
        code_file (file): objeto representando arquivo de saída aberto para escrita.
        state_size (int): tamanho do estado da rede em representação de números de 64 bits.
        network_nodes (list): lista de nós da rede.
        eqs_size (list): lista com o tamanho de cada equação da rede.
        eqs_file_content (list): lista com conteúdo de linhas do arquivo de equações.
        boolean_equations (bool): as equações do código são booleanas.
    """

    code_file.write('   unsigned long long aux['+str(state_size)+'];\n')
    # inicializando aux
    for i in range(state_size):
        code_file.write('   aux['+str(i)+'] = 0;\n')

    # aplicar equações
    if not boolean_equations:
        for i in range(len(network_nodes)) :
            eq = '    aux['+str(i//64)+'] |= (unsigned long long) ( ( '
            line = eqs_file_content[2+i].split('\n')[0].split(' ')
            for y in range(eqs_size[i]):
                eq += '( ( s['+str(int(line[2*y])//64)+'] >> '+str(int(line[2*y])%64)+') % 2 ) * '+str(line[2*y+1])
                if y != eqs_size[i] - 1:
                    eq+=' + '
            eq += ' ) >= '+str(line[len(line)-1])+' ) << '+str(i%64)+';\n'
            code_file.write(eq)
    else:
        #TODO: equações boolenas
        operators = {'and':' && ', 'not':'!', 'or':' || ', '(':'(', ')':')'}
        nodes_dict = {}
        for i in range(len(network_nodes)):
            nodes_dict[network_nodes[i]] = i
        for i in range(len(eqs_file_content)):
            eq = '    aux['+str(i//64)+'] |= (unsigned long long) ( '
            content = eqs_file_content[i].split(' = ')[1].split('\n')[0].split(' ')
            for c in content:
                if c != '':
                    if c in operators:
                        eq += operators[c] + ' '
                    else:
                        eq += '( ( s['+str(int(nodes_dict[c])//64)+'] >> '+str(int(nodes_dict[c])//64)+') % 2 ) '
            eq += ' ) << '+str(i%64)+';\n'
            code_file.write(eq)

    # estado0 e estado1 rebem resultado de aux, andamos 1 passo com as equações da rede
    for i in range(state_size):
        code_file.write('   s['+str(i)+'] = aux['+str(i)+'];\n')

def write_headers(code_file, cpu=False):
    """Escreve no arquivo de saída tlf.cu os headers c++ e CUDA necessários para rodar o programa.

    Args:
        code_file (file): objeto representando arquivo de saída aberto para escrita.
        cpu (bool): código gerado deve ser C++ puro sem funcionalidades CUDA.
    """

    code_file.write('#include <iostream>\n'+
                    '#include <chrono>\n'+
                    '#include <ctime>\n'+
                    '#include <string>\n'+
                    '#include <fstream>\n'+
                    '#include <sstream>\n'+
                    '#include <vector>\n'+
                    '#include <map>\n'+
                    '#include <unordered_map>\n'+
                    '#include <limits>\n'+
                    '#include <stdio.h>\n'+
                    '#include <stdlib.h>\n'+
                    '#include <algorithm>\n'+

                    '#include <random>\n'+
                    '#include <cmath>\n'+
                    '#include <omp.h>\n'+
                    '\nusing namespace std;\n'+
                    'using namespace std::chrono;\n\n')
    if not cpu:
        code_file.write('#include <curand.h>\n'+
                        '#define cudaCheckError() { cudaError_t e=cudaGetLastError(); if(e!=cudaSuccess) { printf("Cuda failure %s:%d: %s",__FILE__,__LINE__,cudaGetErrorString(e)); exit(0); } }\n')


def write_comp_func(code_file, state_size, cpu=False):
    """Escreve no arquivo de saída o código de função de comparação de estados:

    Args:
        code_file (file): objeto representando arquivo de saída aberto para escrita.
        state_size (integer): quantidade de inteiros (64bit) necessários para representar o estado.
        cpu (bool): código gerado deve ser C++ puro sem funcionalidades CUDA.
    """
    if not cpu:
        # função de comparação entre estados kernel
        code_file.write('__device__ bool equals_d(unsigned long long * a, unsigned long long * b) {\n'+
                        '   for (int i = 0; i < '+str(state_size)+'; i++) {\n'+
                        '       if (a[i] != b[i])\n'+
                        '           return false;\n'+
                        '   }\n'+
                        '   return true;\n'+
                        '}\n')

    # função de comparação entre estados host
    code_file.write('bool equals_h(unsigned long long * a, unsigned long long * b) {\n'+
                    '   for (int i = 0; i < '+str(state_size)+'; i++) {\n'+
                    '       if (a[i] != b[i])\n'+
                    '           return false;\n'+
                    '   }\n'+
                    '   return true;\n'+
                    '}\n')

def write_cuda_kernel(code_file, state_size):
    """Escreve no arquivo de saída o código do kernel CUDA para simulação da rede.

    Args:
        code_file (file): objeto representando arquivo de saída aberto para escrita.
        state_size (int): quantidade de inteiros(64bit) necessários para representar o estado.
    """

    code_file.write('__global__ void network_simulation_d(unsigned long long * statef, unsigned long long SIMULATIONS) {\n'+
                    '   unsigned long long tid = threadIdx.x + blockIdx.x*blockDim.x;\n'+
                    '   unsigned long long state0['+ str(state_size) +'], state1['+ str(state_size) +'];\n'+
                    '   if (tid < SIMULATIONS) {\n')
    
    # inicializando estados
    for i in range(state_size):
        code_file.write('       state0['+str(i)+'] = state1['+str(i)+'] = statef[tid*'+ str(state_size) +' + '+ str(i) +'];\n')



    # GPU
    # equações : 
    # estadof[var//64] = 0
    # estadof[var//64] |= ( ( ( (estado0[i//64] >> var) % 2 )*peso + ( (estado0[i//64] >> var) % 2 )*peso  ...) >= lim) << var;
    code_file.write('       do {\n')

    code_file.write('           next_d(state0);\n')

    # andamos 2 passos com estado 1
    code_file.write('           next_d(state1);\n')
    code_file.write('           next_d(state1);\n')

    code_file.write('       } while(!equals_d(state0, state1));\n')

    # salva o estado inicial do atrator na memória global da gpu
    for i in range(state_size) :
        code_file.write('       statef[tid*'+ str(state_size) +' + '+ str(i) +'] = state1['+str(i)+'];\n')
    code_file.write('   }\n'+
                    '}\n')

def generateCudaCode(eqs_file_path, boolean_equations=False, cpu=False, test_both=False):
    """Gera código cuda para simulação da rede utilizando equações com peso em arquivo de saída tlf.cu.
        
    Args:
        weights_file_path (string) : caminho até o arquivo com pesos da rede.
    """

    if test_both:
        cpu = False

    # lendo pesos das redes
    weightsFile = open(eqs_file_path, 'r')
    fileContent = weightsFile.readlines()

    weightsFile.close()

    # definindo valores
    networkNodes = []
    if not boolean_equations:
        networkNodes = fileContent[0].split('\n')[0].split(' ')
    else:
        for l in fileContent:
            networkNodes.append(l.split(' = ')[0].strip())
    networkSize = len(networkNodes)
    weightsSize = []
    if not boolean_equations:
        weightsSize = [ int(x) for x in fileContent[1].split('\n')[0].split(' ')]
    # gerando código da rede
    output_file_name = 'tlf.cu'
    if cpu:
        output_file_name = 'tlf.cpp'
    code_file = open(output_file_name, 'w+')
    
    # headers do código

    write_headers(code_file, cpu)

    # estado é um vetor de inteiros
    # cada bit representa um vértice
    stateSize = networkSize//64 + (networkSize%64 != 0)
    code_file.write('typedef unsigned long long * state;\n')

    # função de comparação entre estados kernel
    write_comp_func(code_file, stateSize, cpu)

    # função host que aplica equações num estado
    code_file.write('void next_h(unsigned long long * s) {\n')

    write_equations(code_file, stateSize, networkNodes, weightsSize, fileContent, boolean_equations)

    code_file.write('}\n')

    # função device que aplica equações da rede em um estado 
    if not cpu:
        code_file.write('__device__ void next_d(unsigned long long * s) {\n')

        # escreve equações
        write_equations(code_file, stateSize, networkNodes, weightsSize, fileContent, boolean_equations)

        code_file.write('}\n')
        write_cuda_kernel(code_file, stateSize)

    # cuda kernel recebe os estados aleatórios inicialmente, simulando N estados até o número de simulações fornecido


    code_file.write('\n')

    # CPU
    # versão cpu do calculo de atratores
    code_file.write('void network_simulation_h(unsigned long long * statef, unsigned long long SIMULATIONS){\n'+
                    '   unsigned long long i;\n'+
                    '   #pragma omp parallel for private (i)\n'+
                    '   for(i = 0; i < SIMULATIONS; i++){\n'+
                    '       unsigned long long state0['+ str(stateSize) +'], state1['+ str(stateSize) +'], aux['+ str(stateSize) +'];\n')

    # inicializando estados
    for i in range(stateSize):
        code_file.write('       state0['+str(i)+'] = state1['+str(i)+'] = statef[i*'+ str(stateSize) +' + '+ str(i) +'];\n')

    code_file.write('       do {\n')

    code_file.write('           next_h(state0);\n')

    # andamos 2 passos com estado 1
    code_file.write('           next_h(state1);\n')
    code_file.write('           next_h(state1);\n')

    code_file.write('       } while(!equals_h(state0, state1));\n')
    # salva o estado inicial do atrator na memória global da gpu
    for i in range(stateSize) :
        code_file.write('       statef[i*'+ str(stateSize) +' + '+ str(i) +'] = state1['+str(i)+'];\n')
    code_file.write('   }\n'+
                    '}\n')

    # função que converte estado para string
    code_file.write("string to_string(unsigned long long * s){\n"+
                    "   string result;\n"+
                    "   stringstream stream;\n"+
                    "   stream << s[0];\n"
                    "   for(int i = 1; i < "+ str(stateSize-1) +"; i++)\n"+
                    "       stream << s[i];\n"+
                    "   stream >> result;\n"+
                    "   return result;\n"+
                    "}\n")

    # função que converte um atrator(vetor de estados) para string
    code_file.write("string to_string(vector<string> atractor){\n"+
                    # gambiarra para imprimir aspas duplas
                    '   if(atractor.size() == 0) return \"[\'\']\";\n'+
                    "   string result = \"[\" + atractor[0];\n"+
                    "   for (int i = 1; i < atractor.size(); i++)\n"+
                    # gambiarra para imprimir aspas duplas
                    '       result += "' + '," + atractor[i];\n'+
                    '   result += \"]\";\n'+
                    "   return result;\n"+
                    "}\n")

    # função que recebe um estado de um atrator e entrega o atrator completo
    # em um vector<string> com representação em string do atrator
    code_file.write("vector<string> getAtractor(unsigned long long * s) {\n"+
                    '   unsigned long long s0['+ str(stateSize) +'], s1['+ str(stateSize) +'], aux['+ str(stateSize) +'];\n'+
                    '   vector<string> atractor; atractor.push_back(to_string(s));\n'+
                    "   for (int i = 0; i < "+str(stateSize)+"; i++){\n"+
                    "       s0[i] = s1[i] = s[i];\n"+
                    "       aux[i] = 0;\n"+
                    "   }\n"+
                    "   while(true) {\n"+
                    "       next_h(s0);\n")
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
    code_file.write('vector<string> complete_atractors(unsigned long long * state_f, unsigned long long SIMULATIONS){\n'+
                    '   vector<string> atractors;\n'
                    '   unordered_map<string, string> state_to_at;\n'+
                    '   unordered_map<string, unsigned long> at_freq;\n'+
                    '   for(unsigned long long i = 0; i < SIMULATIONS; i++) {\n'+
                    '       unsigned long long st['+ str(stateSize) +'];\n'+
                    '       for (size_t j = 0; j < '+ str(stateSize) +'; j++) {\n'+
                    '           st[j] = state_f[i*'+ str(stateSize) +' + j];\n'+
                    '       }\n'+
                    '       string sst = to_string(st);\n'+
                    '       if (state_to_at.count(sst) > 0) {\n'+
                    '           at_freq[state_to_at[sst]]++;\n'+
                    '       } else {\n'+
                    '           vector<string> at = getAtractor(st);\n'+
                    '           string sat = to_string(at);\n'+
                    '           atractors.push_back(sat);\n'+
                    '           for (int j = 0; j < at.size(); j++)\n'+
                    '               state_to_at[at[j]] = sat;\n'+
                    '           at_freq[sat]=1;\n'+
                    '       }\n'+
                    '   }\n'+
                    '   return atractors;\n'+
                    '}\n')

    # função que imprime atratores encontrados num arquivo
    code_file.write('void output_atractors(const vector<string> &atractors) {\n'+
                    '   ofstream atractorsFile;\n'+
                    '   atractorsFile.open("atractors.json");\n'+
                    '   atractorsFile << "{\\n";\n'+
                    '   atractorsFile << "\\"nodes\\" : [";\n')
    for i in range(len(networkNodes)-1):
        code_file.write('   atractorsFile << "\\"'+str(networkNodes[i])+'\\",";\n')
    code_file.write('   atractorsFile << "\\"'+str(networkNodes[len(networkNodes)-1])+'\\"],\\n";\n')

    code_file.write('   atractorsFile << "\\"atractors\\" : [";\n'+
                    '   for (unsigned long long i = 0; i < atractors.size()-1; i++)\n'+
                    '       atractorsFile << atractors[i] <<",";\n'+
                    '   atractorsFile << atractors[atractors.size()-1] <<"]\\n";\n'+
                    '   atractorsFile << "}\\n";\n'+
                    '   atractorsFile.close();'+
                    '}\n')

    # inicializa estados inicias aleatoriamente na gpu
    write_random_initator(code_file, stateSize, cpu)

    # código main, aloca vetores e preencher estados iniciais com números randômicos
    # chama kernel gpu e função cpu para calcular os atratores
    code_file.write('int main(int argc, char **argv) {\n'+
                    '   unsigned long long SIMULATIONS = 0;\n'+    
                    '   std::string argv2 = argv[1];\n'+
                    '   for(int i = 0; i < argv2.size() ; i++)\n'+
                    "       SIMULATIONS += ((unsigned long long)(argv2[i] - '0'))*pow(10,argv2.size()-i-1);\n"+
                    '   cout << "Alocating memory...";\n'+
                    '   unsigned long long * statef_h, * statef_d;\n'+
                    '   statef_h = new unsigned long long[SIMULATIONS*'+ str(stateSize) +'];\n')
    if not cpu:
        code_file.write('   cudaMalloc((unsigned long long **)&statef_d,sizeof(unsigned long long)*SIMULATIONS*'+ str(stateSize) +');\n'+
                        '   cout << "[OK]" << '+repr("\n")+';\n'+
                        '   cudaDeviceProp prop;\n'+
                        '   int device = 0;\n'+
                        '   int threads = 512;\n'+
                        '   #ifdef DEVICE\n'+
                        '       device = DEVICE;\n'+
                        '   #else\n'+
                        '       cudaSetDevice(device);\n'+
                        '       cudaGetDeviceProperties(&prop, device);\n'+
                        '       #ifdef THREADS\n'+
                        '           threads = THREADS;\n'+
                        '       #else\n'+
                        '           threads = prop.maxThreadsPerBlock;\n'+
                        '       #endif\n'+
                        '   #endif\n'+
                        '   cout << "GPU : " << prop.name << '+repr("\n")+';\n'+
                        '   dim3 block(threads);\n'+
                        '   dim3 grid((SIMULATIONS + block.x -1)/block.x);\n'+
                        '   cout << "Number of threads used: " << threads << "  Max allowed :" << prop.maxThreadsDim[0] << '+repr("\n")+';\n'+
                        '   cout << "Number of blocks used: " << grid.x << "  Max allowed :" << prop.maxGridSize[0] << '+repr("\n")+';\n'+
                        '   cout << "Initiating values...";\n'+
                        '   init_rand_d(statef_d, SIMULATIONS);\n'+
                        '   cudaMemcpy(statef_h, statef_d, sizeof(unsigned long long)*SIMULATIONS*'+ str(stateSize) +', cudaMemcpyDeviceToHost);\n'+
                        '   cout << "[OK]" << '+repr("\n")+';\n'+
                        '   cout << "Running Simulation...";\n'+
                        '   auto start_gpu = high_resolution_clock::now();\n'+
                        '   network_simulation_d<<<grid,block>>>(statef_d, SIMULATIONS);\n'+
                        '   cudaCheckError();\n'+
                        '   cudaDeviceSynchronize();\n'+
                        '   cudaMemcpy(statef_h, statef_d, sizeof(unsigned long long)*SIMULATIONS*'+ str(stateSize) +', cudaMemcpyDeviceToHost);\n'+
                        '   auto end_gpu = high_resolution_clock::now();\n')
    if test_both or cpu:
        code_file.write('   auto start_cpu = high_resolution_clock::now();\n'+
                        '   network_simulation_h(statef_h, SIMULATIONS);\n'+
                        '   auto end_cpu = high_resolution_clock::now();\n'+
                        '   cout << "[OK]" << '+repr("\n")+';\n')
    if not cpu:
        code_file.write('   auto dt = duration<double, milli> (end_gpu - start_gpu);\n'+
                        '   cout << "Running Time GPU (ms) : " << dt.count() << '+repr('\n')+';\n')

    if test_both or cpu:
        code_file.write('   auto dt_cpu = duration<double, milli> (end_cpu - start_cpu);\n'+
                        '   cout << "Running Time CPU (ms) : " << dt_cpu.count() << '+repr('\n')+';\n')

    code_file.write('   cout << "Getting atractors found...";\n'+
                    '   vector<string> atratores = complete_atractors(statef_h, SIMULATIONS);\n'
                    '   cout << "[OK]" << '+repr("\n")+';\n'+
                    '   output_atractors(atratores);\n'
                    '   delete [] statef_h;\n')
    if not cpu:
        code_file.write('   cudaFree(statef_d);\n'+
                        '   cudaDeviceReset();\n')
    code_file.write('   return 0;\n'+
                    '}\n')

    code_file.close()

if __name__ == '__main__' :
    """Função main recebe arquivo de rede como parâmetro e gera como saída
    um arquivo tlf.cu para simulação da rede em CUDA.

    Args:
        file (string): caminho até arquivo com equações da rede(boolenas ou de soma de pesos).
        explicit-equations (bool): faz com que as equações sejam diretamente inseridas no código sem a utilização de memória. 
        cpu (bool): código gerado roda somente na CPU, sem funcionalidades CUDA.
    """

    parser = argparse.ArgumentParser(description='Recebe equações TLF '+
            '(soma de pesos) ou boolenas de uma Rede Reguladora e retorna '+
            'como saída um arquivo tlf.cu com código CUDA para '+
            'simulação da rede.'
    )
    parser.add_argument('file', type=str, help='Arquivo contendo equações de pesos ou boolenas')
    parser.add_argument('--boolean-equations', '-b', action='store_true', help='Equações são inseridas diretamente no código sem usar memória')
    parser.add_argument('--cpu', action='store_true', default=False, help='Gera código C++ (.cpp) com openmp')
    parser.add_argument('--test-both', action='store_true', default=False, help='Roda simulações gpu e cpu para comparação de tempo de simulação.' )
    args = parser.parse_args()
    # try :
    eqs_file_path = args.file
    if not os.path.exists(eqs_file_path) :
        print('Arquivo com pesos da rede não foi encontrado!')
    else:
        generateCudaCode(args.file, args.boolean_equations, args.cpu, args.test_both)
