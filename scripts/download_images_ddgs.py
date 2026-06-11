# scripts/download_images_ddg.py

from pathlib import Path
import requests
import shutil
from ddgs import DDGS
from PIL import Image
from io import BytesIO


def download_images(query: str, output_dir: str, max_images: int = 200) -> int:
    temp_folder = Path(output_dir) / "temp"
    temp_folder.mkdir(parents=True, exist_ok=True)
    
    downloaded = 0

    with DDGS() as ddgs:
        results = ddgs.images(
            query,
            max_results=max_images,
            safesearch="moderate",
        )

        for i, result in enumerate(results):
            image_url = result.get("image")
            if not image_url:
                continue

            try:
                response = requests.get(image_url, timeout=10)
                response.raise_for_status()

                img = Image.open(BytesIO(response.content)).convert("RGB")
                file_path = temp_folder / f"{query.replace(' ', '_')}_{downloaded:04d}.jpg"
                img.save(file_path, "JPEG", quality=90)

                downloaded += 1
                print(f"Downloaded {downloaded}: {file_path}")
                
                

            except Exception as e:
                print(f"Skipped result {i}: {e}")

    
    
    # create final folder name
    output_path = Path(output_dir)
    #Extract the name of the output folder, i.e., which animal or object is being downloaded, to write it in the README.md file
    output_path_name = output_path.name
    #The README.md file will be created in the parent folder of the output folder, One file summing up results from all queries will be created in the parent folder of the output folder, and each query will have its own line in the README.md file with the number of downloaded images for that query.
    output_path = output_path.parent
    output_path.mkdir(parents=True, exist_ok=True)
    
    
    print(f"Finished. Downloaded {downloaded} images to {output_path}")
    
    
    readme_path = output_path / "README.md"
    with open(readme_path, "a", encoding="utf-8") as f:
        f.write(f"#### {output_path_name}: Downloaded images: {downloaded}\n")
    
    # move files
    for path in temp_folder.rglob("*"):

        if path.is_file():
            output_path = Path(output_dir)
            destination = Path(output_dir) / path.name

            shutil.move(
                str(path),
                str(destination)
            )
            
    # remove temp folder
    shutil.rmtree(temp_folder)
    return downloaded
    


if __name__ == "__main__":
    TOTALPICT = 0
    
    TOTALPICT += download_images("car photo", "data/raw/not_dog/car", max_images=50)
    TOTALPICT += download_images("cat photo", "data/raw/not_dog/cat", max_images=200)
    TOTALPICT += download_images("empty city photo", "data/raw/not_dog/city", max_images=50)
    TOTALPICT += download_images("empty forest photo", "data/raw/not_dog/forest", max_images=50)
    TOTALPICT += download_images("red fox animal photo", "data/raw/not_dog/fox", max_images=200)
    TOTALPICT += download_images("hyena animal photo", "data/raw/not_dog/hyena", max_images=300)
    TOTALPICT += download_images("wolf animal photo", "data/raw/not_dog/wolf", max_images=300)
    TOTALPICT += download_images("deer animal photo", "data/raw/not_dog/deer", max_images=300)
    TOTALPICT += download_images("horse animal photo", "data/raw/not_dog/horse", max_images=50)
    TOTALPICT += download_images("goat animal photo", "data/raw/not_dog/goat", max_images=50)
    TOTALPICT += download_images("sheep animal photo", "data/raw/not_dog/sheep", max_images=50)
    TOTALPICT += download_images("cow animal photo", "data/raw/not_dog/cow", max_images=50)
    TOTALPICT += download_images("bear animal photo", "data/raw/not_dog/bear", max_images=50)
    TOTALPICT += download_images("human photo", "data/raw/not_dog/human", max_images=50)
    TOTALPICT += download_images("corals photo", "data/raw/not_dog/corals", max_images=50)
    TOTALPICT += download_images("fish animal photo", "data/raw/not_dog/fish", max_images=50)
    TOTALPICT += download_images("dress photo", "data/raw/not_dog/dress", max_images=50)
    TOTALPICT += download_images("empty street photo", "data/raw/not_dog/empty street", max_images=50)
    TOTALPICT += download_images("empty garden", "data/raw/not_dog/empty garden", max_images=50)
    TOTALPICT += download_images("weasel animal photo", "data/raw/not_dog/weasel", max_images=50)
    TOTALPICT += download_images("squirrel animal photo", "data/raw/not_dog/squirrel", max_images=50)
    TOTALPICT += download_images("hamster animal photo", "data/raw/not_dog/hamster", max_images=50)
    TOTALPICT += download_images("rat animal photo", "data/raw/not_dog/rat", max_images=50)
    TOTALPICT += download_images("mouse animal photo", "data/raw/not_dog/mouse", max_images=50)
    TOTALPICT += download_images("cougar animal photo", "data/raw/not_dog/cougar", max_images=50)
    TOTALPICT += download_images("wild boar animal photo", "data/raw/not_dog/wild_boar", max_images=50)
    TOTALPICT += download_images("lynx animal photo", "data/raw/not_dog/lynx", max_images=50)
    TOTALPICT += download_images("opossum animal photo", "data/raw/not_dog/opossum", max_images=50)
    TOTALPICT += download_images("pig animal photo", "data/raw/not_dog/pig", max_images=50)
    TOTALPICT += download_images("tiger animal photo", "data/raw/not_dog/tiger", max_images=50)
    TOTALPICT += download_images("lion animal photo", "data/raw/not_dog/lion", max_images=50)
    
    print(f"Finished. Downloaded Total {TOTALPICT} images")