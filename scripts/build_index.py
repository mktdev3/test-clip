
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict
import time

# srcモジュールへのパスを追加
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.clip_model import CLIPModel
from src.image_encoder import ImageEncoder
from src.vector_store import VectorStore

# ロガーの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Build image index for search')
    parser.add_argument('--image_dir', type=str, default='images',
                        help='Directory containing images to index')
    parser.add_argument('--vector_dir', type=str, default='data/vectors',
                        help='Directory to save vector data')
    parser.add_argument('--metadata_dir', type=str, default='data/metadata',
                        help='Directory to save metadata')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for encoding')
    parser.add_argument('--model_name', type=str, default=None,
                        help='CLIP model name')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cpu/cuda)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # パスの正規化
    image_dir = Path(args.image_dir)
    vector_dir = Path(args.vector_dir)
    metadata_dir = Path(args.metadata_dir)
    
    if not image_dir.exists():
        logger.error(f"Image directory not found: {image_dir}")
        sys.exit(1)
        
    try:
        # モデルの初期化
        logger.info("Initializing CLIP model...")
        clip_model = CLIPModel(model_name=args.model_name, device=args.device)
        encoder = ImageEncoder(clip_model)
        store = VectorStore(vector_dir, metadata_dir)
        
        # 画像ファイルの収集
        logger.info(f"Scanning images in {image_dir}...")
        image_paths = []
        supported_formats = {'.jpg', '.jpeg', '.png'}
        
        for root, _, files in os.walk(image_dir):
            for file in files:
                if Path(file).suffix.lower() in supported_formats:
                    image_paths.append(Path(root) / file)
        
        if not image_paths:
            logger.warning("No images found to index.")
            sys.exit(0)
            
        logger.info(f"Found {len(image_paths)} images.")
        
        # バッチ処理でエンコード
        start_time = time.time()
        vectors = encoder.encode_images_batch(image_paths, batch_size=args.batch_size)
        encoding_time = time.time() - start_time
        
        logger.info(f"Encoding finished in {encoding_time:.2f} seconds.")
        
        # メタデータの作成
        metadata = []
        for path in image_paths:
            # 相対パスを保存する場合
            try:
                rel_path = str(path.resolve().relative_to(Path.cwd()))
            except ValueError:
                # CWD外の場合は絶対パスを使用
                rel_path = str(path.resolve())
            metadata.append({
                'path': rel_path,
                'filename': path.name
            })
            
        # 保存
        logger.info("Saving index...")
        store.save(vectors, metadata)
        logger.info("Index build completed successfully.")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()
