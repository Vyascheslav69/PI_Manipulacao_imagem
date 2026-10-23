import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# CARREGAR E SALVAR IMAGENS
def carregar_imagem(caminho):
    # convert('RGB') garante 3 canais, mesmo se o JPG vier em outro formato.
    img = Image.open(caminho).convert('RGB')
    return np.array(img, dtype=np.float32)

def salvar_imagem(matriz, caminho):
    # Antes de salvar, os valores precisam voltar a ficar entre 0 e 255 e virar números inteiros.
    img = np.clip(matriz, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(caminho)

def escolher_imagem(pasta_imgs):
    # Lista as imagens da pasta
    extensoes = (".jpg", ".jpeg", ".png")
    imagens = sorted(f for f in os.listdir(pasta_imgs) if f.lower().endswith(extensoes))

    print("Imagens disponíveis:")
    for n, nome in enumerate(imagens, start=1):
        print(f"  {n:2d}. {nome}")

    while True:
        escolha = input("Digite o número da imagem (ou o caminho de outra imagem): ").strip()
        if escolha.isdigit() and 1 <= int(escolha) <= len(imagens):
            return os.path.join(pasta_imgs, imagens[int(escolha) - 1])
        if escolha != "" and not escolha.isdigit() and os.path.isfile(escolha):
            return escolha

# Filtros
# FILTRO 1: Converter para Tons de Cinza
# -------------------------------------------------------------
# Tirar a média das 3 cores para todos os pixels ter um único número
def para_cinza(matriz_rgb):
    if len(matriz_rgb.shape) == 2:
        return matriz_rgb.copy()

    r = matriz_rgb[:, :, 0]
    g = matriz_rgb[:, :, 1]
    b = matriz_rgb[:, :, 2]
    return (r + g + b) / 3.0

# FILTRO 2: Blur (Máscara 3x3 de Média)
# -------------------------------------------------------------
# Cada pixel vira a MÉDIA dele com os seus 8 vizinhos:
# somamos os 9 valores e dividimos por 9.
def aplicar_blur(matriz_cinza):
    lins, cols = matriz_cinza.shape
    img_blur = matriz_cinza.copy()
    m = matriz_cinza

    for i in range(1, lins - 1):
        for j in range(1, cols - 1):
            soma = (
                m[i-1, j-1] + m[i-1, j] + m[i-1, j+1] +
                m[i,   j-1] + m[i,   j] + m[i,   j+1] +
                m[i+1, j-1] + m[i+1, j] + m[i+1, j+1]
            )
            img_blur[i, j] = soma / 9.0

    return img_blur

# FILTRO 3: Detecção de Bordas (Sobel com as 4 máscaras)
# -------------------------------------------------------------
# As máscaras de Sobel comparam um lado do pixel com o outro:
#   - Horizontais: linha de CIMA menos linha de BAIXO
#   - Verticais: coluna da ESQUERDA menos coluna da DIREITA
# Cada resultado é limitado entre 0 e 255 (negativo vira 0,
# acima de 255 vira 255), e depois os 4 são somados.
def limitar(valor):
    # Garante que o valor fique entre 0 e 255
    return max(0.0, min(255.0, valor))

def aplicar_sobel(matriz_cinza):
    lins, cols = matriz_cinza.shape
    img_sobel = np.zeros_like(matriz_cinza)
    m = matriz_cinza

    for i in range(1, lins - 1):
        for j in range(1, cols - 1):
            # Máscara Horizontal 1:   1  2  1 /  0  0  0 / -1 -2 -1
            h1 = ( 1*m[i-1, j-1] +  2*m[i-1, j] +  1*m[i-1, j+1] +
                   0*m[i,   j-1] +  0*m[i,   j] +  0*m[i,   j+1] +
                  -1*m[i+1, j-1] + -2*m[i+1, j] + -1*m[i+1, j+1] )

            # Máscara Horizontal 2:  -1 -2 -1 /  0  0  0 /  1  2  1
            h2 = (-1*m[i-1, j-1] + -2*m[i-1, j] + -1*m[i-1, j+1] +
                   0*m[i,   j-1] +  0*m[i,   j] +  0*m[i,   j+1] +
                   1*m[i+1, j-1] +  2*m[i+1, j] +  1*m[i+1, j+1] )

            # Máscara Vertical 1:   1  0 -1 /  2  0 -2 /  1  0 -1
            v1 = ( 1*m[i-1, j-1] +  0*m[i-1, j] + -1*m[i-1, j+1] +
                   2*m[i,   j-1] +  0*m[i,   j] + -2*m[i,   j+1] +
                   1*m[i+1, j-1] +  0*m[i+1, j] + -1*m[i+1, j+1] )

            # Máscara Vertical 2:  -1  0  1 / -2  0  2 / -1  0  1
            v2 = (-1*m[i-1, j-1] +  0*m[i-1, j] +  1*m[i-1, j+1] +
                  -2*m[i,   j-1] +  0*m[i,   j] +  2*m[i,   j+1] +
                  -1*m[i+1, j-1] +  0*m[i+1, j] +  1*m[i+1, j+1] )

            # Cada máscara é limitada a [0, 255] antes da soma
            soma = (limitar(h1) + limitar(h2) +
                    limitar(v1) + limitar(v2))
            img_sobel[i, j] = limitar(soma)

    return img_sobel

# FILTRO 4: Unsharp Mask (Original Cinza - Blur)
# -------------------------------------------------------------
# unsharp = original - blur

def calcular_unsharp(matriz_cinza):
    # os valores negativos são guardados, porque o sharpening precisa deles
    img_blur = aplicar_blur(matriz_cinza)
    return matriz_cinza - img_blur

def aplicar_unsharp(matriz_cinza):
    return np.clip(calcular_unsharp(matriz_cinza), 0, 255)

# FILTRO 5: Sharpening (Original Cinza + Unsharp)
# -------------------------------------------------------------
#     sharpening = original + unsharp
#                = original + (original - blur)

def aplicar_sharpening(matriz_cinza):
    unsharp = calcular_unsharp(matriz_cinza)
    sharpening = matriz_cinza + unsharp
    return np.clip(sharpening, 0, 255)


# PROGRAMA PRINCIPAL
if __name__ == "__main__":
    pasta_script = os.path.dirname(os.path.abspath(__file__))

    if len(sys.argv) > 1:
        caminho_imagem = sys.argv[1]
    else:
        caminho_imagem = escolher_imagem(os.path.join(pasta_script, "imgs"))

    try:
        img_original = carregar_imagem(caminho_imagem)

        # Aplica as 5 transformações
        img_cinza = para_cinza(img_original)
        img_blur = aplicar_blur(img_cinza)
        img_sobel = aplicar_sobel(img_cinza)
        img_unsharp = aplicar_unsharp(img_cinza)
        img_sharpening = aplicar_sharpening(img_cinza)

        # Salva cada resultado como imagem na pasta
        pasta_saida = os.path.join(pasta_script, "resultados")
        os.makedirs(pasta_saida, exist_ok=True)
        nome_base = os.path.splitext(os.path.basename(caminho_imagem))[0]

        resultados = {
            "1_cinza": img_cinza,
            "2_blur": img_blur,
            "3_sobel": img_sobel,
            "4_unsharp": img_unsharp,
            "5_sharpening": img_sharpening,
        }
        for nome, matriz in resultados.items():
            caminho_saida = os.path.join(pasta_saida, f"{nome_base}_{nome}.png")
            salvar_imagem(matriz, caminho_saida)
            print(f"Salvo: {caminho_saida}")

        # Monta a exibição em grade
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))

        axes[0, 0].imshow(img_original.astype(np.uint8))
        axes[0, 0].set_title("Original (Colorida)")

        axes[0, 1].imshow(img_cinza, cmap='gray', vmin=0, vmax=255)
        axes[0, 1].set_title("1. Tons de Cinza")

        axes[0, 2].imshow(img_blur, cmap='gray', vmin=0, vmax=255)
        axes[0, 2].set_title("2. Blur")

        axes[1, 0].imshow(img_sobel, cmap='gray', vmin=0, vmax=255)
        axes[1, 0].set_title("3. Sobel (Bordas)")

        axes[1, 1].imshow(img_unsharp, cmap='gray', vmin=0, vmax=255)
        axes[1, 1].set_title("4. Unsharp Mask")

        axes[1, 2].imshow(img_sharpening, cmap='gray', vmin=0, vmax=255)
        axes[1, 2].set_title("5. Sharpening")

        for ax in axes.ravel():
            ax.axis('off')

        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print(f"Erro: Não foi possível encontrar a imagem '{caminho_imagem}'. Verifique o caminho da imagem!")
