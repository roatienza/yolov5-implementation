"""
Download COCO 2017 dataset automatically.
IMPORTANT: Always use /data directory in sandbox environment.
"""

import os
import urllib.request
import zipfile
import shutil
from pathlib import Path
from tqdm import tqdm

# IMPORTANT: Always use /data directory in sandbox environment
DATA_ROOT = '/data/memory/coco'  # Sandbox-mounted data directory


class COCODatasetDownloader:
    """Downloads COCO 2017 dataset with standard splits."""
    
    def __init__(self, output_dir: str = DATA_ROOT):
        """
        Args:
            output_dir: Root directory for dataset storage (default: /data/memory/coco)
        """
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / 'images'
        self.annotations_dir = self.output_dir / 'annotations'
        
        # COCO 2017 dataset URLs
        self.image_urls = {
            'train2017': 'http://images.cocodataset.org/zips/train2017.zip',
            'val2017': 'http://images.cocodataset.org/zips/val2017.zip',
            'test2017': 'http://images.cocodataset.org/zips/test2017.zip'
        }
        
        self.annotation_urls = {
            'trainval2017': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
            'test-dev2017': 'http://images.cocodataset.org/annotations/image_info_test-dev2017.zip'
        }
    
    def _download_file(self, url: str, dest_path: Path) -> None:
        """Download a file with progress bar."""
        if dest_path.exists():
            print(f"File already exists: {dest_path}")
            return
        
        print(f"Downloading from {url} to {dest_path}")
        
        def reporthook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, (downloaded / total_size) * 100)
                desc = f"Downloaded {downloaded / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB ({percent:.1f}%)"
                pbar.update(downloaded - pbar.n)
                pbar.desc = desc
        
        pbar = tqdm(total=0, unit='B', unit_scale=True, desc="Downloading")
        
        try:
            urllib.request.urlretrieve(url, dest_path, reporthook)
            pbar.total = pbar.n  # Update total to actual size
            pbar.close()
            print(f"Download complete: {dest_path}")
        except Exception as e:
            pbar.close()
            raise e
    
    def _extract_zip(self, zip_path: Path, extract_to: Path) -> None:
        """Extract a ZIP file."""
        if not zip_path.exists():
            raise FileNotFoundError(f"ZIP file not found: {zip_path}")
        
        print(f"Extracting {zip_path} to {extract_to}")
        extract_to.mkdir(parents=True, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        
        print(f"Extraction complete: {extract_to}")
    
    def download_images(self, split: str = 'all') -> None:
        """
        Download COCO images.
        
        Args:
            split: 'train', 'val', 'test', or 'all'
        """
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        splits_to_download = []
        if split in ['all', 'train']:
            splits_to_download.append('train2017')
        if split in ['all', 'val']:
            splits_to_download.append('val2017')
        if split in ['all', 'test']:
            splits_to_download.append('test2017')
        
        for split_name in splits_to_download:
            url = self.image_urls[split_name]
            zip_filename = f"{split_name}.zip"
            zip_path = self.output_dir / zip_filename
            extract_dir = self.images_dir / split_name
            
            # Skip if already extracted
            if extract_dir.exists() and len(list(extract_dir.glob('*.jpg'))) > 0:
                print(f"Images already extracted for {split_name}")
                continue
            
            self._download_file(url, zip_path)
            self._extract_zip(zip_path, self.images_dir)
            
            # Remove ZIP after extraction to save space
            if zip_path.exists():
                zip_path.unlink()
                print(f"Removed ZIP file: {zip_path}")
    
    def download_annotations(self, split: str = 'all') -> None:
        """
        Download COCO annotations.
        
        Args:
            split: 'trainval' or 'test-dev' or 'all'
        """
        self.annotations_dir.mkdir(parents=True, exist_ok=True)
        
        annotations_to_download = []
        if split in ['all', 'trainval']:
            annotations_to_download.append('trainval2017')
        if split in ['all', 'test-dev']:
            annotations_to_download.append('test-dev2017')
        
        for ann_name in annotations_to_download:
            url = self.annotation_urls[ann_name]
            zip_filename = f"{ann_name}.zip"
            zip_path = self.output_dir / zip_filename
            
            # Skip if already downloaded
            if zip_path.exists():
                print(f"Annotation file already exists: {zip_path}")
                continue
            
            self._download_file(url, zip_path)
            self._extract_zip(zip_path, self.annotations_dir)
    
    def download_all(self) -> None:
        """Download complete COCO 2017 dataset."""
        print("=" * 60)
        print("Downloading COCO 2017 Dataset")
        print(f"Output directory: {self.output_dir}")
        print("=" * 60)
        
        print("\n[1/2] Downloading images...")
        self.download_images('all')
        
        print("\n[2/2] Downloading annotations...")
        self.download_annotations('all')
        
        print("\n" + "=" * 60)
        print("COCO 2017 dataset download complete!")
        print("=" * 60)
        self._print_dataset_info()
    
    def _print_dataset_info(self) -> None:
        """Print dataset statistics."""
        print("\nDataset Structure:")
        print(f"  Images: {self.images_dir}")
        print(f"  Annotations: {self.annotations_dir}")
        
        for split in ['train2017', 'val2017', 'test2017']:
            split_dir = self.images_dir / split
            if split_dir.exists():
                img_count = len(list(split_dir.glob('*.jpg')))
                print(f"    {split}: {img_count} images")
        
        for ann_file in self.annotations_dir.glob('*.json'):
            print(f"    {ann_file.name}")


def download_coco_dataset(output_dir: str = DATA_ROOT, split: str = 'all') -> COCODatasetDownloader:
    """
    Downloads COCO 2017 dataset with standard splits.
    ALWAYS operates in sandbox /data directory.
    
    Args:
        output_dir: Root directory for dataset storage (default: /data/memory/coco)
        split: 'train', 'val', 'test', or 'all'
    
    Structure created in /data/memory/coco/:
        /data/memory/coco/
        ├── images/
        │   ├── train2017/
        │   ├── val2017/
        │   └── test2017/
        └── annotations/
            ├── instances_train2017.json
            ├── instances_val2017.json
            └── image_info_test-dev2017.json
    
    Returns:
        COCODatasetDownloader instance
    """
    downloader = COCODatasetDownloader(output_dir)
    if split == 'all':
        downloader.download_all()
    else:
        downloader.download_images(split)
        downloader.download_annotations(split)
    return downloader


if __name__ == '__main__':
    # Example usage
    download_coco_dataset()
