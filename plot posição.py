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


# FAZER A FUNÇÃO DEVOLVER TRUE OU FALSE CONFORME A ONTINUIDADE DO PONTO
# def pontos_proximos_e_novos(historico_de_posicoes, a, PONTO):
def pontos_proximos_e_novos(historico_de_posicoes, a, distancia):
    pontos_proximos = []
    pontos_novos = []

    # Função para calcular a distância Euclidiana entre dois pontos (x1, y1) e (x2, y2)
    def calcular_distancia(p1, p2):
        return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

    # Iterar pelos pontos em historico_de_posicoes[a+1]
    for ponto_b in historico_de_posicoes[a + 1]:
        encontrou_proximo = False
        # Verificar se há um ponto em historico_de_posicoes[a] dentro da distância mínima
        for ponto_a in historico_de_posicoes[a]:
            if calcular_distancia(ponto_a, ponto_b) <= distancia:
                pontos_proximos.append(ponto_b)
                encontrou_proximo = True
                break
        # Se não encontrou nenhum ponto próximo, adicionar à lista de pontos novos
        if not encontrou_proximo:
            pontos_novos.append(ponto_b)

    return pontos_proximos, pontos_novos

def adicionar_ao_historico(lista, historico_de_posicoes, a):

    if len(historico_de_posicoes) >= a:
        historico_de_posicoes.pop(0)

    historico_de_posicoes.append(lista)

def plot_points(historico_de_posicoes):
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

def plotar_latas(historico_de_posicoes, a, distancia_entre_latas):

    latas_confirmadas, latas_novas = pontos_proximos_e_novos(historico_de_posicoes, a, distancia_entre_latas)
    
    for latas in latas_confirmadas:
        plt.scatter(latas[0], latas[1], c='blue', label='Pontos')

    for latas in latas_novas:
        plt.scatter(latas[0], latas[1], c='green', label='Pontos')


    plt.title('centro das latas')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(0, 1040)
    plt.ylim(992, 0) 
    plt.grid(True)
    plt.show()

distancia_entre_latas = 20

frame = 0

# plotar_latas(historico_de_posicoes, frame, distancia_entre_latas)

plot_points(historico_de_posicoes)

print(f'Frame 1: {len(historico_de_posicoes[0])}')
print(f'Frame 2: {len(historico_de_posicoes[1])}')
