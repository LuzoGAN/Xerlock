import cv2
import numpy as np
import os
from pathlib import Path
from datetime import datetime
import concurrent.futures
import time

class BatchImageEnhancer:
    def __init__(self, input_folder, output_folder, scale_factor=2):
        """
        Processa todas as imagens de uma pasta no modo conservador
        
        Args:
            input_folder: Pasta com imagens originais
            output_folder: Pasta para salvar imagens processadas
            scale_factor: 1.5, 2, ou 3 (recomendo 2 para conservador)
        """
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.scale_factor = scale_factor
        
        # Criar pasta de saída se não existir
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        # Extensões suportadas
        self.supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        # Estatísticas
        self.stats = {
            'processed': 0,
            'skipped': 0,
            'errors': 0,
            'total_size_original': 0,
            'total_size_processed': 0,
            'start_time': None,
            'end_time': None
        }
    
    def get_image_files(self):
        """Retorna lista de todos os arquivos de imagem na pasta de entrada"""
        image_files = []
        
        for ext in self.supported_extensions:
            image_files.extend(self.input_folder.glob(f'*{ext}'))
            image_files.extend(self.input_folder.glob(f'*{ext.upper()}'))
        
        return sorted(image_files)
    
    def conservative_enhance(self, image_path, output_path):
        """
        Processamento conservador de uma única imagem
        - Redução de ruído suave
        - Upscale tradicional (não IA)
        - Preservação de textura natural
        """
        try:
            # Ler imagem
            img = cv2.imread(str(image_path))
            if img is None:
                return False, "Não foi possível ler a imagem"
            
            original_h, original_w = img.shape[:2]
            
            # 1. Redução de ruído BEM SUAVE (preserva textura)
            # Apenas em imagens muito pequenas ou com muito ruído
            if original_w < 300 or original_h < 300:
                img = cv2.fastNlMeansDenoisingColored(
                    img, None,
                    h=6,  # Valor BAIXO
                    hColor=6,
                    templateWindowSize=7,
                    searchWindowSize=21
                )
            
            # 2. Upscale com método tradicional de alta qualidade
            new_width = int(original_w * self.scale_factor)
            new_height = int(original_h * self.scale_factor)
            
            # Escolher método de interpolação baseado no tamanho
            if self.scale_factor <= 2:
                interpolation = cv2.INTER_CUBIC  # Bom equilíbrio
            else:
                interpolation = cv2.INTER_LANCZOS4  # Melhor para > 2x
            
            img_resized = cv2.resize(img, (new_width, new_height), interpolation=interpolation)
            
            # 3. Adicionar textura sutil para evitar aspecto "liso/pintado"
            # Apenas se a imagem for suficientemente grande
            if new_width > 300 and new_height > 300:
                img_resized = self._add_natural_texture(img_resized)
            
            # 4. Nitidez MÍNIMA apenas se necessário
            if original_w < 500:  # Imagens pequenas podem precisar
                img_resized = self._apply_minimal_sharpening(img_resized)
            
            # 5. Determinar formato de saída
            output_ext = output_path.suffix.lower()
            
            # Parâmetros de compressão
            if output_ext in {'.jpg', '.jpeg'}:
                # JPEG com alta qualidade
                params = [cv2.IMWRITE_JPEG_QUALITY, 92]
            elif output_ext == '.webp':
                # WebP com alta qualidade
                params = [cv2.IMWRITE_WEBP_QUALITY, 90]
            else:
                # PNG, BMP, TIFF - sem compressão lossy
                params = []
            
            # Salvar
            cv2.imwrite(str(output_path), img_resized, params)
            
            return True, "Sucesso"
            
        except Exception as e:
            return False, str(e)
    
    def _add_natural_texture(self, img, strength=0.008):
        """
        Adiciona textura granular sutil para manter aparência natural
        
        strength: 0.005-0.015 (quanto maior, mais textura)
        """
        h, w = img.shape[:2]
        
        # Criar ruído granular
        noise = np.random.randn(h, w, 3).astype(np.float32)
        
        # Suavizar levemente o ruído
        noise = cv2.GaussianBlur(noise, (0, 0), 0.5)
        
        # Aplicar apenas em áreas uniformes (não em bordas)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, None, iterations=1)
        edges_mask = edges.astype(np.float32) / 255.0
        
        # Inverter para áreas suaves
        smooth_areas = 1 - edges_mask
        
        # Aplicar ruído
        img_float = img.astype(np.float32)
        for c in range(3):
            noise_channel = noise[:,:,c] * strength * 255 * smooth_areas
            img_float[:,:,c] += noise_channel
        
        return np.clip(img_float, 0, 255).astype(np.uint8)
    
    def _apply_minimal_sharpening(self, img, intensity=0.1):
        """
        Aplica nitidez mínima para melhorar detalhes sem criar artefatos
        """
        if intensity <= 0:
            return img
        
        # Unsharp mask muito suave
        blurred = cv2.GaussianBlur(img, (0, 0), 1.0)
        details = cv2.addWeighted(img, 1.0 + intensity, blurred, -intensity, 0)
        
        # Aplicar apenas em bordas detectadas
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges_mask = edges.astype(np.float32) / 255.0
        
        # Misturar
        result = img.copy()
        for c in range(3):
            result[:,:,c] = img[:,:,c] * (1 - edges_mask*intensity) + \
                           details[:,:,c] * (edges_mask*intensity)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def process_all(self, max_workers=None, preserve_structure=True):
        """
        Processa todas as imagens da pasta de entrada
        
        Args:
            max_workers: Número de threads paralelas (None = automático)
            preserve_structure: Se True, mantém subpastas
        """
        print("=" * 70)
        print("BATCH IMAGE ENHANCER - MODO CONSERVADOR")
        print("=" * 70)
        print(f"Pasta de origem: {self.input_folder}")
        print(f"Pasta de destino: {self.output_folder}")
        print(f"Fator de escala: {self.scale_factor}x")
        print("-" * 70)
        
        self.stats['start_time'] = datetime.now()
        
        # Coletar todas as imagens
        all_images = []
        
        if preserve_structure:
            # Manter estrutura de subpastas
            for file_path in self.input_folder.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                    # Calcular caminho relativo
                    rel_path = file_path.relative_to(self.input_folder)
                    output_path = self.output_folder / rel_path
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    all_images.append((file_path, output_path))
        else:
            # Apenas arquivos da pasta raiz
            for file_path in self.get_image_files():
                output_path = self.output_folder / file_path.name
                all_images.append((file_path, output_path))
        
        total_images = len(all_images)
        
        if total_images == 0:
            print("❌ Nenhuma imagem encontrada na pasta de origem!")
            return
        
        print(f"Encontradas {total_images} imagem(ns) para processar")
        print("-" * 70)
        
        # Processar imagens
        processed_count = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submeter todas as tarefas
            future_to_image = {
                executor.submit(self.conservative_enhance, inp, out): (inp, out)
                for inp, out in all_images
            }
            
            # Processar resultados conforme completam
            for future in concurrent.futures.as_completed(future_to_image):
                inp_path, out_path = future_to_image[future]
                
                try:
                    success, message = future.result()
                    processed_count += 1
                    
                    if success:
                        self.stats['processed'] += 1
                        
                        # Calcular tamanhos
                        orig_size = inp_path.stat().st_size / 1024  # KB
                        out_size = out_path.stat().st_size / 1024 if out_path.exists() else 0
                        
                        self.stats['total_size_original'] += orig_size
                        self.stats['total_size_processed'] += out_size
                        
                        print(f"✅ [{processed_count}/{total_images}] {inp_path.name}")
                        print(f"   Tamanho: {orig_size:.1f}KB → {out_size:.1f}KB")
                    else:
                        self.stats['skipped'] += 1
                        print(f"⚠️  [{processed_count}/{total_images}] {inp_path.name}")
                        print(f"   Pulado: {message}")
                        
                except Exception as e:
                    self.stats['errors'] += 1
                    print(f"❌ [{processed_count}/{total_images}] {inp_path.name}")
                    print(f"   Erro: {str(e)}")
                
                # Progresso
                progress = (processed_count / total_images) * 100
                print(f"   Progresso: {progress:.1f}%")
                print()
        
        self.stats['end_time'] = datetime.now()
        self._print_summary()
    
    def _print_summary(self):
        """Imprime resumo do processamento"""
        print("=" * 70)
        print("RESUMO DO PROCESSAMENTO")
        print("=" * 70)
        
        duration = self.stats['end_time'] - self.stats['start_time']
        minutes, seconds = divmod(duration.total_seconds(), 60)
        
        print(f"Tempo total: {int(minutes)}min {seconds:.1f}s")
        print(f"Imagens processadas: {self.stats['processed']}")
        print(f"Imagens puladas: {self.stats['skipped']}")
        print(f"Erros: {self.stats['errors']}")
        print()
        
        if self.stats['processed'] > 0:
            avg_orig = self.stats['total_size_original'] / self.stats['processed']
            avg_proc = self.stats['total_size_processed'] / self.stats['processed']
            
            print(f"Tamanho médio original: {avg_orig:.1f} KB")
            print(f"Tamanho médio processado: {avg_proc:.1f} KB")
            print(f"Razão de tamanho: {avg_proc/avg_orig:.2f}x")
            print()
        
        print(f"Pasta de saída: {self.output_folder}")
        print("=" * 70)

def main():
    """Função principal com interface interativa"""
    print("🤖 BATCH IMAGE ENHANCER - MODO CONSERVADOR")
    print()
    
    # Configurar caminhos
    input_folder = input("Digite a pasta de origem com as imagens: ").strip()
    
    # Sugerir pasta de destino padrão
    default_output = str(Path(input_folder).parent / "imagens_melhoradas")
    output_folder = input(f"Digite a pasta de destino [Enter para '{default_output}']: ").strip()
    
    if not output_folder:
        output_folder = default_output
    
    # Fator de escala
    print("\nSelecione o fator de escala:")
    print("1. 1.5x (mais conservador, mantém textura)")
    print("2. 2.0x [RECOMENDADO] (bom equilíbrio)")
    print("3. 3.0x (maior, mas pode perder qualidade)")
    
    scale_choice = input("Escolha (1-3) [2]: ").strip()
    
    scale_map = {'1': 1.5, '2': 2.0, '3': 3.0}
    scale_factor = scale_map.get(scale_choice, 2.0)
    
    # Processamento paralelo
    use_parallel = input("\nUsar processamento paralelo para velocidade? (s/n) [s]: ").strip().lower()
    max_workers = None if use_parallel in ('s', '') else 1
    
    # Manter estrutura
    preserve_structure = input("\nManter estrutura de subpastas? (s/n) [s]: ").strip().lower()
    preserve_structure = preserve_structure in ('s', '')
    
    print("\n" + "=" * 70)
    print("CONFIGURAÇÃO:")
    print(f"Origem: {input_folder}")
    print(f"Destino: {output_folder}")
    print(f"Escala: {scale_factor}x")
    print(f"Paralelo: {'Sim' if max_workers is None else 'Não'}")
    print(f"Manter estrutura: {'Sim' if preserve_structure else 'Não'}")
    print("=" * 70)
    
    confirm = input("\nIniciar processamento? (s/n) [s]: ").strip().lower()
    
    if confirm not in ('s', ''):
        print("Processamento cancelado.")
        return
    
    print("\nIniciando processamento...")
    print()
    
    # Criar e executar processador
    enhancer = BatchImageEnhancer(input_folder, output_folder, scale_factor)
    
    try:
        enhancer.process_all(max_workers=max_workers, preserve_structure=preserve_structure)
        
        # Perguntar se quer abrir a pasta de destino
        open_folder = input("\nAbrir pasta de destino? (s/n) [n]: ").strip().lower()
        if open_folder == 's':
            os.startfile(output_folder)  # Windows
            
    except Exception as e:
        print(f"\n❌ Erro durante o processamento: {str(e)}")

# ============================================================================
# VERSÃO SIMPLES PARA USO DIRETO
# ============================================================================

def processar_pasta_simples(pasta_origem, pasta_destino, escala=2.0):
    """
    Função simples para processar uma pasta rapidamente
    
    Args:
        pasta_origem: Caminho da pasta com imagens originais
        pasta_destino: Caminho da pasta para salvar resultados
        escala: Fator de aumento (1.5, 2.0, ou 3.0)
    """
    print(f"Processando {pasta_origem} → {pasta_destino}")
    
    enhancer = BatchImageEnhancer(pasta_origem, pasta_destino, escala)
    enhancer.process_all(max_workers=4, preserve_structure=True)
    
    print("✅ Processamento concluído!")

# ============================================================================
# EXECUTAR
# ============================================================================

if __name__ == "__main__":
    # Opção 1: Interface interativa
    main()
    
    # Opção 2: Uso direto (descomente e ajuste os caminhos)
    # processar_pasta_simples(
    #     pasta_origem=r"C:\Users\luzo.neto\Downloads\fotos_originais",
    #     pasta_destino=r"C:\Users\luzo.neto\Downloads\fotos_melhoradas",
    #     escala=2.0
    # )
