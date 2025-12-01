import pandas as pd
import requests
import os
from urllib.parse import urlparse
import time

# Carregar o arquivo Excel
df = pd.read_excel('/content/seguidores_extraidos.xlsx', sheet_name='seguidores_extraidos')

# Criar diretório para salvar as imagens
os.makedirs('fotos_perfil', exist_ok=True)

def download_image(url, username, full_name, index):
    """Baixa uma imagem e salva com o formato username-full_name.jpg"""
    if pd.isna(url) or url == '':
        print(f"[{index}] URL vazio para {username}")
        return
    
    try:
        # Limpar nome de arquivo
        safe_username = str(username).replace('/', '_').replace('\\', '_').strip()
        safe_full_name = str(full_name).replace('/', '_').replace('\\', '_').strip() if not pd.isna(full_name) else ''
        
        # Criar nome do arquivo
        if safe_full_name and safe_full_name != 'nan':
            filename = f"{safe_username}-{safe_full_name}.jpg"
        else:
            filename = f"{safe_username}.jpg"
        
        # Caminho completo
        filepath = os.path.join('fotos_perfil', filename)
        
        # Verificar se arquivo já existe
        if os.path.exists(filepath):
            # Adicionar número se arquivo já existir
            base, ext = os.path.splitext(filename)
            counter = 1
            while os.path.exists(os.path.join('fotos_perfil', f"{base}_{counter}{ext}")):
                counter += 1
            filepath = os.path.join('fotos_perfil', f"{base}_{counter}{ext}")
        
        # Fazer download da imagem
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Salvar imagem
        with open(filepath, 'wb') as f:
            f.write(response.content)
        
        print(f"[{index}] Baixado: {filename}")
        
        # Delay para evitar sobrecarregar o servidor
        time.sleep(0.5)
        
    except requests.exceptions.RequestException as e:
        print(f"[{index}] Erro ao baixar {username}: {str(e)}")
    except Exception as e:
        print(f"[{index}] Erro inesperado com {username}: {str(e)}")

# Baixar todas as imagens
print(f"Iniciando download de {len(df)} imagens...")
print("-" * 50)

for index, row in df.iterrows():
    download_image(
        url=row['profile_pic'],
        username=row['username'],
        full_name=row['full_name'],
        index=index
    )

print("-" * 50)
print("Download concluído!")
print(f"Imagens salvas na pasta: {os.path.abspath('fotos_perfil')}")
