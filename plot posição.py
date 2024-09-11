import numpy as np
import matplotlib.pyplot as plt
import math

historico_de_posicoes = [[(370, 79), (306, 301), (784, 201), (643, 179), (246, 62), (625, 878), (696, 822), (177, 531), (309, 136), (343, 212), (670, 674), (531, 695), (237, 822), (540, 768), (309, 743), (145, 687), (755, 64), (855, 30), (894, 384), (214, 598), (899, 120), (805, 511), (183, 215), (223, 673), (448, 605), (262, 218), (299, 668), (711, 233), (255, 527), (379, 674), (420, 401), (620, 795), (758, 446), (411, 539), (523, 226), (864, 195), (602, 58), (334, 531), (448, 70), (505, 900), (524, 618), (781, 871), (481, 309), (442, 233), (524, 62), (847, 818), (129, 320), (292, 593), (342, 392), (539, 969), (606, 471), (484, 151), (989, 455), (678, 74), (408, 162), (680, 448), (320, 820), (745, 650), (755, 723), (836, 743), (768, 287), (893, 648), (376, 467), (426, 893), (758, 576), (872, 293), (215, 465), (452, 476), (464, 753), (136, 608), (574, 398), (683, 747), (223, 955), (379, 953), (562, 842), (944, 32), (262, 384), (345, 889), (606, 723), (453, 678), (977, 140), (481, 825), (928, 928), (946, 706), (207, 324), (640, 268), (567, 553), (718, 154), (567, 299), (299, 465), (487, 542), (567, 157), (992, 783), (815, 121), (836, 596), (915, 774), (771, 797), (912, 468), (946, 587), (384, 315), (498, 409), (837, 445), (176, 24), (817, 345), (958, 330), (265, 890), (396, 824), (528, 478), (924, 846), (93, 545), (139, 453), (598, 648), (815, 670), (299, 959), (881, 537), (705, 897), (862, 896), (368, 600), (640, 536), (642, 343), (720, 512), (36, 499), (681, 599), (387, 749), (723, 354), (855, 968), (230, 746), (321, 2)], [(755, 57), (642, 170), (862, 186), (309, 289), (177, 526), (989, 448), (60, 271), (248, 55), (501, 892), (784, 192), (448, 598), (975, 130), (340, 379), (398, 818), (540, 761), (624, 789), (146, 
684), (486, 301), (412, 531), (443, 221), (320, 812), (345, 204), (680, 440), (696, 814), (301, 952), (417, 390), (523, 218), (465, 747), (670, 667), (184, 208), (334, 526), (406, 152), (448, 60), (137, 602), (255, 521), (232, 742), (237, 815), (711, 224), (293, 589), (387, 745), (684, 742), (452, 467), (678, 65), (223, 665), (312, 129), (484, 
143), (262, 209), (379, 667), (487, 536), (780, 865), (221, 949), (600, 51), (815, 340), (847, 811), (379, 947), (625, 870), (894, 371), (129, 311), (299, 662), (531, 686), 
(756, 570), (758, 442), (371, 68), (831, 589), (836, 439), (526, 54), (608, 462), (376, 461), (683, 593), (994, 875), (767, 277), (899, 112), (642, 331), (483, 820), (214, 592), (384, 311), (524, 611), (759, 718), (370, 593), (893, 642), (205, 320), (496, 401), (912, 462), (639, 259), (815, 112), (946, 700), (537, 962), (265, 368), (567, 546), 
(962, 329), (818, 662), (803, 506), (295, 458), (558, 836), (864, 286), (306, 736), (884, 528), (990, 775), (40, 496), (265, 884), (718, 145), (567, 290), (571, 146), (853, 
23), (771, 937), (595, 640), (104, 540), (345, 883), (924, 839), (214, 456), (527, 471), (608, 715), (705, 890), (944, 580), (836, 737), (936, 922), (136, 453), (455, 955), 
(914, 767), (859, 889), (427, 887), (720, 506), (173, 17), (853, 961), (944, 24), (774, 789), (574, 386), (721, 334), (742, 642), (455, 670)]]

distancia_entre_latas = 20

def eh_lata_nova(frame, posicao):

    # Função para calcular a distância Euclidiana entre dois pontos (x1, y1) e (x2, y2)
    def calcular_distancia(p1, p2):
        return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
    
    # Iterar pelos pontos em historico_de_posicoes[a+1]
    encontrou_proximo = False

    # Verificar se há um ponto em historico_de_posicoes[a] dentro da distância mínima
    for posicoes_anteriores in historico_de_posicoes[frame -1]:
        if calcular_distancia(posicoes_anteriores, posicao) <= distancia_entre_latas:
            return True

        # Se não encontrou nenhum ponto próximo, adicionar à lista de pontos novos
        if not encontrou_proximo:
            return False

def adicionar_ao_historico(lista, a):

    if len(historico_de_posicoes) >= a:
        historico_de_posicoes.pop(0)

    historico_de_posicoes.append(lista)

def plot_points():
    frame = 0
    lata = 0
    latas = []
    for frames in historico_de_posicoes:
        lata = 0
        if frame == 0:
            for ponto in frames:    
                plt.scatter(ponto[0], ponto[1], c='blue', label='Pontos')
                lata += 1
                
        if frame == 1:
            for ponto in frames:    
                plt.scatter(ponto[0], ponto[1], c='green', label='Pontos')
                lata += 1
        latas.append(lata)
        frame += 1


    plt.title('centro das latas')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(0, 1040)  # Definindo o limite do eixo x entre 0 e 1040
    plt.ylim(992, 0) 
    # plt.legend()
    plt.grid(True)
    plt.show()

    print(f'Latas contadas: {latas}')

def plotar_latas():
    ultimo_frame = len(historico_de_posicoes) -1
    print(f'ultimo frame: {ultimo_frame}')
    # Caso seja o primeiro Frame
    if ultimo_frame < 0:
        
        print('Sem Frames')
        return

    if ultimo_frame == 0:
        print("frame 1")
        for lata in historico_de_posicoes[ultimo_frame]:
            
            plt.scatter(lata[0], lata[1], c='blue', label='Pontos')
            return

    print(f'frame {ultimo_frame+1}')
    for lata in historico_de_posicoes[ultimo_frame]:
        
        foi_plotado = 0
        if eh_lata_nova(ultimo_frame, lata):
            plt.scatter(lata[0], lata[1], c='green', label='Pontos')
            foi_plotado += 1
        else:
            plt.scatter(lata[0], lata[1], c='blue', label='Pontos')
            foi_plotado += 1
    
    print(f'latas plotadas: {foi_plotado}')

    plt.title('Posições das latas')
    plt.xlim(0, 1040)
    plt.ylim(992, 0) 
    plt.grid(True)
    plt.show()

plotar_latas()

# plot_points(historico_de_posicoes)
