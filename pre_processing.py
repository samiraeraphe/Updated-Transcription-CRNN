import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import shutil
import lmdb
import io
from tkinter import *
from deslant_img import deslant_img
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk, UnidentifiedImageError

def rotacionar_imagem(imagem, angulo):
    """
    Rotaciona a imagem pelo ângulo especificado com fundo branco para áreas vazias.
    
    :param imagem: Imagem a ser rotacionada.
    :param angulo: Ângulo de rotação em graus. Valores positivos para rotações anti-horárias,
                   valores negativos para rotações horárias.
    :return: Imagem rotacionada com fundo branco nas áreas vazias.
    """
    altura, largura = imagem.shape[:2]
    ponto_central = (largura // 2, altura // 2)
    
    # Obter a matriz de rotação usando o ponto central da imagem e o ângulo fornecido
    matriz_rotacao = cv2.getRotationMatrix2D(ponto_central, angulo, 1.0)
    
    # Calcular as novas dimensões da imagem para evitar cortes
    cos = np.abs(matriz_rotacao[0, 0])
    sen = np.abs(matriz_rotacao[0, 1])
    
    nova_largura = int((altura * sen) + (largura * cos))
    nova_altura = int((altura * cos) + (largura * sen))
    
    matriz_rotacao[0, 2] += (nova_largura / 2) - ponto_central[0]
    matriz_rotacao[1, 2] += (nova_altura / 2) - ponto_central[1]

    # Aplicar a rotação com borda branca e interpolação por vizinho mais próximo
    imagem_rotacionada = cv2.warpAffine(imagem, matriz_rotacao, (nova_largura, nova_altura),
                                        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=(255))
    return imagem_rotacionada

def adicionar_linhas_overlay(imagem, color=(128, 128, 128), transparencia=0.5):
    overlay = imagem.copy()
    altura = imagem.shape[0]
    espaco_entre_linhas = altura // 15  # Aumentando a frequência das linhas
    for y in range(espaco_entre_linhas, altura, espaco_entre_linhas):
        cv2.line(overlay, (0, y), (imagem.shape[1], y), color, 2)
    cv2.addWeighted(overlay, transparencia, imagem, 1 - transparencia, 0, imagem)
    return imagem

def ajustar_rotacao(imagem):
    root = Tk()
    root.title("Ajuste de Rotação da Imagem")

    angulo = IntVar(value=0)
    confirmado = False

    def atualizar_imagem():
        nonlocal img_tk
        rotacionada = rotacionar_imagem(imagem, angulo.get())
        rotacionada_com_overlay = adicionar_linhas_overlay(rotacionada.copy())
        image_rotated = Image.fromarray(cv2.cvtColor(rotacionada_com_overlay, cv2.COLOR_BGR2RGB))
        img_tk = ImageTk.PhotoImage(image=image_rotated)
        label_imagem.config(image=img_tk)

    def aumentar_angulo():
        angulo.set(angulo.get() + 1)
        atualizar_imagem()

    def diminuir_angulo():
        angulo.set(angulo.get() - 1)
        atualizar_imagem()

    def confirmar():
        nonlocal confirmado
        confirmado = True
        root.quit()
        root.destroy()

    image = Image.fromarray(cv2.cvtColor(imagem, cv2.COLOR_GRAY2RGB))
    img_tk = ImageTk.PhotoImage(image=image) 
    label_imagem = Label(root, image=img_tk)
    label_imagem.pack()

    frame_botoes = Frame(root)
    btn_diminuir = Button(frame_botoes, text="-1 Grau", command=diminuir_angulo)
    btn_diminuir.pack(side=LEFT)
    btn_aumentar = Button(frame_botoes, text="+1 Grau", command=aumentar_angulo)
    btn_aumentar.pack(side=LEFT)
    btn_confirmar = Button(frame_botoes, text="Confirmar", command=confirmar)
    btn_confirmar.pack(side=LEFT)

    frame_botoes.pack()

    root.protocol("WM_DELETE_WINDOW", confirmar)
    root.mainloop()

    if confirmado:
        return angulo.get()
    else:
        return None


def preprocessamento_e_thresholding(path_imagem):
    # Carregar imagem
    imagem = cv2.imread(path_imagem)
    img = cv2.fastNlMeansDenoisingColored(imagem, None, 5, 5, 7, 21)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(72, 256))
    equalized = clahe.apply(gray)
    imagem_denoised_post = cv2.fastNlMeansDenoising(equalized, None, 20, 3, 50)

    # Aplicar thresholding adaptativo
    binarizada = imagem_denoised_post

    # Definir a largura da borda que você deseja eliminar
    borda = 5  # Isso define uma borda de 10 pixels
    color = 255

    # Eliminar ruídos nas bordas definindo os pixels para preto
    binarizada[:borda, :] = color  # Superior
    binarizada[-borda:, :] = color  # Inferior
    binarizada[:, :borda] = color  # Esquerda
    binarizada[:, -borda:] = color  # Direita

    angulo = ajustar_rotacao(binarizada)

    rotacionada = rotacionar_imagem(binarizada, angulo)

    # Visualizar o resultado do thresholding e remoção de ruído
    cv2.imshow('original', imagem)
    cv2.imshow('Thresholding Adaptativo', binarizada)
    cv2.imshow('Rotacao', rotacionada)
    # cv2.imshow('Thresholding Adaptativo 2', binarizada2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Opcional: Retornar a imagem processada para uso posterior
    return rotacionada

def filtrar_linhas_sem_texto(linhas_segmentadas, limiar_densidade=500):
    linhas_com_texto = []
    for linha in linhas_segmentadas:
        # Calcular a densidade de pixels da linha
        densidade_pixels = np.count_nonzero(linha)
        # print(densidade_pixels)
        # Verificar se a linha contém texto com base no limiar de densidade
        if densidade_pixels > limiar_densidade:
            linhas_com_texto.append(linha)
    return linhas_com_texto

def adicionar_borda_preta(imagem, top=15, bottom=15, left=10, right=10):
    """
    Adiciona uma borda preta ao redor da imagem.
    
    :param imagem: Imagem binarizada e invertida (texto branco, fundo preto).
    :param top: Espessura da borda superior.
    :param bottom: Espessura da borda inferior.
    :param left: Espessura da borda esquerda.
    :param right: Espessura da borda direita.
    :return: Imagem com borda preta adicionada.
    """
    borda_preta = cv2.copyMakeBorder(imagem, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return borda_preta


def limpar_ruidos(linha, kernel_size=(5,5)):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    linha_limpa = cv2.morphologyEx(linha, cv2.MORPH_OPEN, kernel)
    return linha_limpa

def remover_ruido_por_densidade(img_binaria, limiar_densidade=10, altura_limite=0.1):
    # Criar uma cópia da imagem binarizada
    img_sem_ruido = img_binaria.copy()

    # Altura da imagem
    altura = img_binaria.shape[0]

    # Limites para considerar regiões próximas das bordas
    limite_superior = altura * altura_limite
    limite_inferior = altura * (1 - altura_limite)

    # Encontrar os componentes conectados na imagem binarizada
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_binaria, connectivity=8)

    # Determinar o componente principal (maior componente excluindo o fundo)
    principal_component_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1

    # Iterar sobre os componentes conectados
    for label in range(1, num_labels):  # Começa em 1 para ignorar o componente de fundo
        if label == principal_component_label:
            continue  # Ignorar o componente principal

        # A área do componente
        area = stats[label, cv2.CC_STAT_AREA]

        # Posição y do componente
        y_position = centroids[label][1]

        # Condição adicional para verificar se está próximo das bordas superior ou inferior
        if y_position < limite_superior or y_position > limite_inferior:
            densidade = np.count_nonzero(labels == label)
            if densidade < limiar_densidade or area < limiar_densidade:
                img_sem_ruido[labels == label] = 0
        else:
            # Em regiões centrais, ser mais permissivo com a remoção
            if area < limiar_densidade:
                img_sem_ruido[labels == label] = 0

    return img_sem_ruido

def segmentar_linhas_em_palavras(linhas_filtradas, margem_inicio=5, margem_fim=5, limiar_espaco=10, min_colunas_espaco=3):
    """
    Segmenta cada linha de texto em palavras, baseando-se na projeção vertical.
    
    :param linhas_filtradas: Lista de linhas segmentadas, cada uma sendo uma matriz de pixels.
    :param margem_inicio: Margem adicional adicionada ao início de cada palavra segmentada.
    :param margem_fim: Margem adicional adicionada ao fim de cada palavra segmentada.
    :param limiar_espaco: Limiar para identificar espaços entre palavras.
    :param min_colunas_espaco: Número mínimo de colunas consecutivas abaixo do limiar de espaço necessárias para considerar um espaço.
    :return: Lista de listas, onde cada sublista contém as palavras segmentadas de uma linha.
    """
    palavras_por_linha = []

    for linha in linhas_filtradas:
        linha = adicionar_borda_preta(linha)
        linha = remover_ruido_por_densidade(linha)
        # cv2.imshow("resultado", linha)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        projecao_vertical = np.sum(linha, axis=0)
        
        palavras = []
        em_palavra = False
        colunas_espaco = 0  # Contador para as colunas abaixo do limiar de espaço
        inicio_palavra = 0  # Inicialização do índice de início da palavra
        
        for i, valor in enumerate(projecao_vertical):
            if valor > limiar_espaco:
                if not em_palavra:
                    inicio_palavra = i
                    em_palavra = True
                colunas_espaco = 0  # Resetar o contador de colunas de espaço ao encontrar pixel acima do limiar
            else:
                colunas_espaco += 1
                if em_palavra and colunas_espaco >= min_colunas_espaco:
                    fim_palavra = i - colunas_espaco + 1  # Ajustar o fim da palavra para o início da sequência de espaços
                    ajustado_inicio_palavra = max(0, inicio_palavra - margem_inicio)
                    ajustado_fim_palavra = min(linha.shape[1], fim_palavra + margem_fim)
                    palavra = linha[:, ajustado_inicio_palavra:ajustado_fim_palavra]
                    palavras.append(palavra)
                    em_palavra = False
                    colunas_espaco = 0  # Resetar o contador de colunas de espaço após segmentar a palavra

        # Para garantir que a última palavra seja adicionada se ainda estivermos em uma palavra no final da linha
        if em_palavra:
            ajustado_fim_palavra = min(linha.shape[1], len(projecao_vertical) + margem_fim)
            palavra = linha[:, inicio_palavra:ajustado_fim_palavra]
            palavras.append(palavra)

        palavras_por_linha.append(palavras)
    
    return palavras_por_linha

def extrair_linhas_com_margem(img_path, margem_inicio, margem_fim, threshold, limiar_projecao):
    # Binarizar a imagem
    binarizada = preprocessamento_e_thresholding(img_path)

    # Projeção horizontal para identificar linhas
    projecao_horizontal = np.sum(binarizada, axis=1)

    # Identificação de linhas baseada na projeção
    linhas = []
    em_linha = False
    for i, valor in enumerate(projecao_horizontal):
        if valor <= threshold and not em_linha:
            # Início de uma nova linha de texto
            inicio_linha = i
            em_linha = True
        elif valor > threshold and em_linha:
            # Final de uma linha de texto
            fim_linha = i
            # Ajustar o corte tanto no início quanto no final da linha
            ajustado_inicio_linha = max(0, inicio_linha - margem_inicio)  # Evita índice negativo
            ajustado_fim_linha = min(len(projecao_horizontal), fim_linha + margem_fim)  # Evita índice maior que a altura da imagem
            # Verificar se o valor da projeção horizontal excede o limiar durante a segmentação da linha
            if np.any(projecao_horizontal[ajustado_inicio_linha:ajustado_fim_linha] < limiar_projecao):
                # Se o limiar for excedido, descartar a segmentação atual e continuar para a próxima linha
                em_linha = False
                continue
            linhas.append((ajustado_inicio_linha, ajustado_fim_linha))
            em_linha = False

    # Para garantir que a última linha seja adicionada
    if em_linha:
        ajustado_fim_linha = min(len(projecao_horizontal), i + margem_fim)
        linhas.append((ajustado_inicio_linha, ajustado_fim_linha))

    # Segmentar a imagem em linhas individuais
    linhas_segmentadas = [binarizada[linha[0]:linha[1], :] for linha in linhas]

    # linhas_filtradas = filtrar_linhas_sem_texto(linhas_segmentadas)

    linhas_sem_angulacao = []

    # Mostrar as linhas segmentadas
    for idx, linha in enumerate(linhas_segmentadas):
        # linha = remover_ruido_por_densidade(linha)
        res = deslant_img(linha, 'grid', -2, 2, 20, 255)
        linhas_sem_angulacao.append(res.img)
        plt.imshow(res.img, cmap='gray', vmin=0, vmax=255)
        plt.title('Deslanted')
        # plt.figure(figsize=(10, 1))
        # plt.imshow(linha, cmap='gray')
        # plt.title(f'Linha {idx + 1}')
        plt.show()

    # Visualizar a projeção horizontal
    # plt.plot(projecao_horizontal)
    # plt.xlabel('Linha')
    # plt.ylabel('Soma de Pixels')
    # plt.title('Projeção Horizontal')
    # plt.show()

    # palavras_das_linhas = segmentar_linhas_em_palavras(linhas_sem_angulacao)

    # Visualizar as palavras segmentadas de uma linha específica
    # for i, linha in enumerate(palavras_das_linhas[3]):  # Visualizar palavras da primeira linha segmentada
    #     plt.figure(figsize=(2, 2))
    #     plt.imshow(linha, cmap='gray')
    #     plt.title(f'Palavra {i + 1}')
    #     plt.show()

    return linhas_sem_angulacao

# Chamada da função com parâmetros personalizados


def resize_images(images, target_size):
    resized_images = []
    for img in images:
        # Redimensiona a imagem para o tamanho alvo usando o método de interpolação cv2.INTER_AREA
        resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)
        resized_images.append(resized_img)
    return resized_images



# for idx, linha in enumerate(vetor_de_linhas):
#         plt.imshow(linha, cmap='gray')
#         plt.title(f'Linha {idx + 1}')
#         plt.show()

def save_line_images(image_filename, line_images, base_dir):
    # Extraia o número do texto e da página usando expressão regular
    match = re.match(r"texto(\d+)_pagina(\d+)\.png", image_filename)
    if not match:
        raise ValueError("Nome do arquivo não segue o formato 'textoX_paginaY.png'")

    text_num, page_num = match.groups()

    # Certifique-se de que o diretório base existe
    os.makedirs(base_dir, exist_ok=True)

    # Caminho do diretório para esta página específica dentro do texto
    page_dir = os.path.join(base_dir, f'texto{text_num}_pagina{page_num}')
    os.makedirs(page_dir, exist_ok=True)

    # Salvar cada linha da página com numeração única
    for line_index, line_img in enumerate(line_images):
        filename = f'texto{text_num}_pagina{page_num}_linha{line_index:04d}.png'
        filepath = os.path.join(page_dir, filename)
        cv2.imwrite(filepath, line_img)

    print(f"Images for text {text_num} page {page_num} saved in {page_dir}")

def review_images_in_directory(directory):
    def load_images():
        # Carrega todas as imagens PNG em ordem numérica baseada no nome
        files = [file for file in os.listdir(directory) if file.endswith('.png')]
        return sorted(files, key=lambda x: int(re.findall(r"linha(\d+)\.png", x)[0]))

    def update_image(img_path):
        img = Image.open(img_path)
        photo = ImageTk.PhotoImage(img)
        image_label.config(image=photo)
        image_label.image = photo

    def keep_image():
        nonlocal index
        kept_files.append(image_files[index])
        index += 1
        if index < len(image_files):
            update_image(os.path.join(directory, image_files[index]))
        else:
            root.quit()

    def discard_image():
        nonlocal index
        os.remove(os.path.join(directory, image_files[index]))
        index += 1
        if index < len(image_files):
            update_image(os.path.join(directory, image_files[index]))
        else:
            root.quit()

    image_files = load_images()
    kept_files = []
    index = 0

    if image_files:
        root = Tk()
        root.title("Image Reviewer")
        image_label = Label(root)
        image_label.pack()
        btn_keep = Button(root, text="Keep", command=keep_image)
        btn_keep.pack(side="left", expand=True)
        btn_discard = Button(root, text="Discard", command=discard_image)
        btn_discard.pack(side="right", expand=True)
        update_image(os.path.join(directory, image_files[index]))
        root.mainloop()
    
    root.destroy()
    return kept_files  # Retorna lista de imagens mantidas

def renumber_images(directory, kept_files):
    # Verificar se a lista de arquivos mantidos está vazia
    if not kept_files:
        print("Nenhuma imagem para renomear.")
        return

    # Extrair o número do texto e da página do nome do diretório usando expressão regular
    match = re.search(r"texto(\d+)_pagina(\d+)", directory)
    if not match:
        raise ValueError("O diretório não segue o formato 'textoX_paginaY'")

    text_num, page_num = match.groups()

    # Verificar se a sequência numérica dos 4 últimos dígitos dos nomes das imagens está correta
    expected_index = 0
    need_renaming = False
    for file in kept_files:
        index_match = re.search(r"(\d{4})\.png$", file)
        if index_match:
            actual_index = int(index_match.group(1))
            if actual_index != expected_index:
                need_renaming = True
                break
            expected_index += 1

    # Renomear arquivos mantidos, se necessário
    if need_renaming:
        for index, file in enumerate(kept_files, start=0):
            old_path = os.path.join(directory, file)
            new_filename = f'texto{text_num}_pagina{page_num}_linha{index:04d}.png'
            new_path = os.path.join(directory, new_filename)

            # Renomeia o arquivo
            os.rename(old_path, new_path)

        print("Imagens renomeadas com sucesso.")
    else:
        print("A numeração das imagens está correta. Nenhuma renomeação necessária.")

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def process_images(original_path, processed_path):
    # Garantir que o diretório de saída existe
    os.makedirs(processed_path, exist_ok=True)

    # Percorrer todas as imagens no caminho original, ordenadas naturalmente
    filenames = sorted(os.listdir(original_path), key=natural_keys)
    print("--------------Iniciando pré-processamento das imagens--------------\n")
    for filename in filenames:
        if filename.endswith('.png'):
            print(filename)
            # Extrair linhas com margem
            vetor_de_linhas = extrair_linhas_com_margem(os.path.join(original_path, filename), margem_inicio=13, margem_fim=13, threshold=127000, limiar_projecao=0)

            # Redimensionar imagens
            vetor_de_linhas = resize_images(vetor_de_linhas, (800, 50))

            # Salvar as linhas das imagens
            save_line_images(filename, vetor_de_linhas, processed_path)

            # Revisar e renomear as imagens
            page_dir = os.path.join(processed_path, os.path.splitext(filename)[0])
            kept_files = review_images_in_directory(page_dir)
            renumber_images(page_dir, kept_files)

            # Mover a imagem original para um subdiretório de imagens processadas
            processed_subdir = os.path.join(original_path, "processed")
            os.makedirs(processed_subdir, exist_ok=True)
            shutil.move(os.path.join(original_path, filename), processed_subdir)

            # Caminho completo para o arquivo .txt
            caminho_arquivo = os.path.join(original_path, "ultima_imagem_processada.txt")
            
            # Salva o nome da última imagem processada no arquivo .txt
            with open(caminho_arquivo, "w") as arquivo_txt:
                arquivo_txt.write(filename)

            print("--------------Próxima página--------------\n")

def initialize_lmdb(lmdb_path, map_size=100e6):
    """ Inicializa a base de dados LMDB com um tamanho de mapa especificado. """
    env = lmdb.open(lmdb_path, map_size=int(map_size))
    env.close()
    print("LMDB database initialized at", lmdb_path)

def format_filename(filename):
    """ Formata o nome do arquivo para incluir texto e página com dois dígitos, mantendo a linha como está. """
    
    # Extrair a parte da linha do nome do arquivo original
    file_parts = filename.split('_')
    texto_num = file_parts[0][5:].zfill(2)
    pagina_num = file_parts[1][6:].zfill(2)
    linha_part = file_parts[2]  # Assumindo que a estrutura é sempre 'linhaZ.png'
    
    # Retorna o nome do arquivo formatado corretamente
    return f"texto{texto_num}_pagina{pagina_num}_{linha_part}"

def process_and_add_images_to_lmdb(source_dir, lmdb_path):
    """ Processa imagens em subdiretórios e adiciona à base de dados LMDB, movendo pastas processadas. """
    env = lmdb.open(lmdb_path, map_size=int(30e6))

    # Verificar se a pasta 'added_pages' já existe e criar se não existir
    processed_dir = os.path.join(source_dir, "added_pages")
    os.makedirs(processed_dir, exist_ok=True)

    # Lista e ordena subdiretórios baseando-se nos números extraídos
    subdirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d)) and d != 'added_pages']
    subdirs.sort(key=lambda x: (int(x.split('_')[0][5:]), int(x.split('_')[1][6:])))

    for subdir in subdirs:
        subdir_path = os.path.join(source_dir, subdir)

        with env.begin(write=True) as txn:
            for filename in os.listdir(subdir_path):
                if filename.lower().endswith('png'):
                    file_path = os.path.join(subdir_path, filename)
                    image_data = Image.open(file_path)
                    
                    # Converter a imagem para bytes
                    byte_array = io.BytesIO()
                    image_data.save(byte_array, format='JPEG')
                    image_bytes = byte_array.getvalue()

                    # Gerar o nome formatado da chave
                    formatted_filename = format_filename(filename)

                    # Armazenar no LMDB
                    txn.put(formatted_filename.encode(), image_bytes)
                    
        # Mover pasta processada
        shutil.move(subdir_path, os.path.join(processed_dir, subdir))
        
    env.close()
    print("All images have been added and directories moved.")

def read_lmdb_images(lmdb_path):
    env = lmdb.open(lmdb_path, readonly=True)
    
    # Cria a janela GUI fora do contexto da transação LMDB
    root = Tk()
    root.title("Image Viewer")

    label = Label(root)
    label.pack()

    # Adiciona um botão para avançar para a próxima imagem
    next_image_var = IntVar(value=0)  # Variável de controle
    btn_next = Button(root, text="Próxima Imagem", command=lambda: next_image_var.set(1))
    btn_next.pack()

    with env.begin() as txn:
        cursor = txn.cursor()

        # Ordenar as chaves
        sorted_keys = sorted([key for key, _ in cursor if not key.startswith(b'label_')], key=lambda x: x.decode())

        for key in sorted_keys:
            value = txn.get(key)
            image = Image.open(io.BytesIO(value))
            photo = ImageTk.PhotoImage(image)
            
            # Busca o rótulo correspondente
            label_key = b'label_' + key
            text = txn.get(label_key)
            if text:
                print(f"Label for {key.decode()}: {text.decode()}")  # Exibe o rótulo no terminal
            
            label.config(image=photo)
            label.image = photo  # Manter referência!

            # Reseta a variável e espera o usuário clicar em "Próxima Imagem"
            next_image_var.set(0)
            root.wait_variable(next_image_var)

    # Fecha a janela após sair do loop
    root.destroy()
    env.close()


def list_image_keys_and_labels(lmdb_path):
    # Abrir o ambiente LMDB no caminho especificado
    env = lmdb.open(lmdb_path, readonly=True)

    # Criar uma transação somente leitura
    with env.begin() as txn:
        # Criar um cursor para iterar sobre a base de dados
        cursor = txn.cursor()

        # Iterar sobre as chaves na base de dados
        for key, value in cursor:
            # Decodificar a chave
            key_str = key.decode('utf-8')
            
            # Verificar se a chave é para uma imagem ou para um rótulo
            if key_str.startswith('label_'):
                # Se for uma chave de rótulo, decodificar o valor como texto UTF-8
                label = value.decode('utf-8')
                image_key = key_str[6:]  # Remover o prefixo 'label_'
                print(f'Imagem: {image_key}, Rótulo: {label}')
            else:
                # Se for uma chave de imagem, imprimir que não tem rótulo associado
                print(f'Imagem: {key_str}, Rótulo: sem rótulo')

    # Fechar o ambiente LMDB
    env.close()

def label_images_gui(lmdb_path):
    """ Cria uma GUI para adicionar rótulos às imagens na base de dados LMDB. """
    root = Tk()
    root.title("Image Labeling")

    # Estado de navegação
    current_index = [0]
    keys = []
    labels = {}
    
    def refresh_keys_labels():
        # Abrir o ambiente LMDB temporariamente para ler as chaves
        env = lmdb.open(lmdb_path, readonly=True)
        with env.begin() as txn:
            cursor = txn.cursor()
            keys[:] = [key for key, _ in cursor if key.startswith(b'texto')]
            labels.update({key: txn.get(b'label_' + key[6:]).decode('utf-8') if txn.get(b'label_' + key[6:]) else "" for key in keys})
        env.close()
        current_index[0] = next((i for i, key in enumerate(keys) if not labels[key]), 0)

    # Carregar a imagem e verificar se tem rótulo
    def load_image():
        env = lmdb.open(lmdb_path, readonly=True)
        with env.begin() as txn:
            key = keys[current_index[0]]
            data = txn.get(key)
        env.close()
        
        image = Image.open(io.BytesIO(data))
        photo = ImageTk.PhotoImage(image)
        current_label = labels[key]
        return photo, key.decode(), current_label

    # Atualizar a imagem no label
    def update_image():
        photo, key, current_label = load_image()
        image_label.config(image=photo)
        image_label.image = photo  # Manter referência
        filename_label.config(text=f"File: {key}")
        current_label_label.config(text=f"Current Label: {current_label}")

    # Funções de navegação
    def next_image():
        if current_index[0] < len(keys) - 1:
            current_index[0] += 1
            update_image()

    def previous_image():
        if current_index[0] > 0:
            current_index[0] -= 1
            update_image()

    # Função para adicionar rótulo
    def add_label():
        key = keys[current_index[0]]
        label_key = b'label_' + key[6:]
        label = simpledialog.askstring("Input", "Enter label for the image:", parent=root, initialvalue=labels[key])
        if label is not None:
            env = lmdb.open(lmdb_path, readonly=False)
            with env.begin(write=True) as txn:
                txn.put(label_key, label.encode())
                labels[key] = label
                current_label_label.config(text=f"Current Label: {label}")
                messagebox.showinfo("Info", "Label updated successfully!")
            env.close()

    # Carregar chaves e rótulos na inicialização
    refresh_keys_labels()

    # Elementos da GUI
    image_label = Label(root)
    image_label.pack()

    filename_label = Label(root, text="")
    filename_label.pack()

    current_label_label = Label(root, text="")
    current_label_label.pack()

    btn_previous = Button(root, text="Previous", command=previous_image)
    btn_previous.pack(side=LEFT)

    btn_next = Button(root, text="Next", command=next_image)
    btn_next.pack(side=RIGHT)

    btn_label = Button(root, text="Add Label", command=add_label)
    btn_label.pack(side=BOTTOM)

    update_image()
    root.mainloop()

def update_lmdb_with_text(lmdb_path, filter_string, text_file_path):
    # Abre o arquivo de texto e lê todas as linhas
    with open(text_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Abre o ambiente LMDB
    env = lmdb.open(lmdb_path, readonly=False, max_dbs=0)
    with env.begin(write=True) as txn:
        # Cria um cursor para percorrer a base de dados
        cursor = txn.cursor()
        # Filtra as chaves que contêm a string específica e estão em ordem lexicográfica
        keys_filtered = [key for key in cursor.iternext(keys=True, values=False) if filter_string.encode() in key]
        
        # Verifica se o número de linhas no arquivo corresponde ao número de chaves filtradas
        if len(keys_filtered) != len(lines):
            print("Aviso: O número de linhas no arquivo de texto não corresponde ao número de chaves encontradas.")
        
        # Associa cada linha do texto a uma chave correspondente
        for key, line in zip(keys_filtered, lines):
            # Atualiza o valor na base de dados em uma chave separada para rótulos
            label_key = b'label_' + key
            txn.put(label_key, line.strip().encode('utf-8'))  # strip() remove espaços/novas linhas extras

    # Fecha o ambiente LMDB
    env.close()
    print("Atualização concluída com sucesso!")

def delete_entries_by_key_string(lmdb_path, string_to_match):
    """ Deleta entradas na LMDB cujas chaves contêm uma string especificada. """
    env = lmdb.open(lmdb_path, readonly=False)  # Abre o ambiente com permissão de escrita

    with env.begin(write=True) as txn:
        cursor = txn.cursor()
        # Deletar as chaves
        for key, _ in cursor.iternext(keys=True, values=False):
            if string_to_match.encode() in key:
                if txn.delete(key):
                    print(f"Deleted key: {key.decode()}")
                else:
                    print(f"Failed to delete key: {key.decode()}")

    env.close()  # Fecha o ambiente LMDB

def count_entries(lmdb_path):
    # Abrir o ambiente LMDB no caminho especificado
    env = lmdb.open(lmdb_path, readonly=True)

    # Inicializar uma contagem de entradas
    num_entries = 0

    # Criar uma transação somente leitura
    with env.begin() as txn:
        # Criar um cursor para iterar sobre a base de dados
        cursor = txn.cursor()

        # Iterar sobre as chaves na base de dados e contar cada uma
        for _ in cursor:
            num_entries += 1

    # Fechar o ambiente LMDB
    env.close()

    # Retornar o número total de entradas na base de dados
    return num_entries

def remove_entries_without_label(lmdb_path):
    # Abre o ambiente LMDB
    env = lmdb.open(lmdb_path, readonly=False)

    with env.begin(write=True) as txn:
        # Cria um cursor para percorrer a base de dados
        cursor = txn.cursor()

        # Itera sobre as chaves na base de dados
        for key, value in cursor:
            # Verifica se a chave não começa com "label_"
            if not key.startswith(b'label_'):
                # Verifica se a chave tem um rótulo associado
                label_key = b'label_' + key
                if txn.get(label_key) is None:
                    # Se não houver rótulo associado, exclui a entrada
                    txn.delete(key)
                    print(f"Entrada sem rótulo associado removida: {key.decode('utf-8')}")

    # Fecha o ambiente LMDB
    env.close()
    print("Remoção de entradas concluída.")

def check_entries_without_label(lmdb_path):
    # Abre o ambiente LMDB
    env = lmdb.open(lmdb_path, readonly=True)

    with env.begin() as txn:
        # Cria um cursor para percorrer a base de dados
        cursor = txn.cursor()

        # Variável para acompanhar se há alguma entrada sem rótulo
        has_entry_without_label = False

        # Itera sobre as chaves na base de dados
        for key, _ in cursor:
            # Verifica se a chave não começa com "label_"
            if not key.startswith(b'label_'):
                # Verifica se há um rótulo associado à chave
                label_key = b'label_' + key
                if txn.get(label_key) is None:
                    # Se não houver rótulo associado, define a flag como True
                    has_entry_without_label = True
                    print(f"Entrada sem rótulo associado encontrada: {key.decode('utf-8')}")

        # Verifica se alguma entrada foi encontrada sem rótulo
        if not has_entry_without_label:
            print("Nenhuma entrada sem rótulo associado encontrada.")

    # Fecha o ambiente LMDB
    env.close()

# check_entries_without_label("lmdb")

vetor_de_linhas = extrair_linhas_com_margem("original_imgs/texto1_pagina1.png", margem_inicio=13, margem_fim=13, threshold=127000, limiar_projecao=10000)

# vetor_de_linhas = resize_images(vetor_de_linhas, (800, 50))

# save_line_images("texto1_pagina1.png", vetor_de_linhas, "processed_imgs/")

# kept_imgs = review_images_in_directory("processed_imgs/texto2_pagina1")

# renumber_images("processed_imgs/texto2_pagina1", kept_imgs)


process_images("original_imgs/", "processed_imgs/")

# initialize_lmdb("lmdb")

# process_and_add_images_to_lmdb("processed_imgs", "lmdb")

# read_lmdb_images("lmdb")

# label_images_gui("lmdb")

# list_image_keys_and_labels("lmdb")

# update_lmdb_with_text("lmdb", "texto30_pagina03", "img_transcription/texto30_pagina3.txt")

# delete_entries_by_key_string("lmdb", "label_label")

# print(count_entries("lmdb"))

# remove_entries_without_label("lmdb")

def rename_images_in_directory(directory):
    """
    Percorre todas as imagens em todas as pastas de um diretório e renomeia as imagens
    de modo que os números representados por X e Y tenham dois dígitos.
    """
    # Regex para encontrar nomes de arquivos no formato "textoX_paginaY_linhaZ.png"
    pattern = re.compile(r"texto(\d+)_pagina(\d+)_linha(\d+)\.png")

    for root, _, files in os.walk(directory):
        for file in files:
            match = pattern.match(file)
            if match:
                x, y, z = match.groups()
                new_filename = f"texto{int(x):02}_pagina{int(y):02}_linha{z}.png"
                old_filepath = os.path.join(root, file)
                new_filepath = os.path.join(root, new_filename)
                
                # Renomear o arquivo
                os.rename(old_filepath, new_filepath)
                print(f"Renamed: {old_filepath} -> {new_filepath}")

# Exemplo de uso
# directory_path = 'processed_imgs/added_pages'
# rename_images_in_directory(directory_path)

def update_lmdb_with_images_from_directory(lmdb_path, images_directory):
    """
    Percorre todas as imagens de todas as pastas de um caminho e substitui as chaves correspondentes na LMDB.
    
    Args:
        lmdb_path (str): Caminho para a base de dados LMDB.
        images_directory (str): Caminho para o diretório contendo as imagens.
    """
    def resize_lmdb(env):
        """Função para aumentar o tamanho do mapa da LMDB."""
        current_size = env.info()['map_size']
        new_size = current_size * 2
        env.set_mapsize(new_size)
        print(f"Map size increased from {current_size} to {new_size}")

    # Abrir o ambiente LMDB com um tamanho de mapa inicial grande
    env = lmdb.open(lmdb_path, map_size=int(100e6), readonly=False, max_dbs=0)

    try:
        # Percorrer todas as pastas e arquivos de imagens no diretório especificado
        for root, _, files in os.walk(images_directory):
            for file in files:
                if file.lower().endswith('.png'):
                    file_path = os.path.join(root, file)
                    
                    # Ler a imagem e convertê-la para bytes
                    with open(file_path, 'rb') as f:
                        image_bytes = f.read()

                    updated = False
                    while not updated:
                        try:
                            # Atualizar a chave correspondente na LMDB
                            with env.begin(write=True) as txn:
                                # Verificar se a chave existe na LMDB
                                if txn.get(file.encode()):
                                    txn.put(file.encode(), image_bytes)
                                    print(f"Updated key: {file}")
                                else:
                                    print(f"Key not found in LMDB: {file}")
                            updated = True
                        except lmdb.MapFullError:
                            resize_lmdb(env)

    finally:
        # Fechar o ambiente LMDB
        env.close()
        print("LMDB updated with images from directory.")

# Exemplo de uso
# lmdb_path = 'lmdb'
# images_directory = 'processed_imgs/added_pages'
# update_lmdb_with_images_from_directory(lmdb_path, images_directory)