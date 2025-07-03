from icrawler.builtin import GoogleImageCrawler
from PIL import Image
import os

# Save images to a folder called 'nutrition_labels'
crawler = GoogleImageCrawler(storage={"root_dir": "nutrition_labels"})

# Download 50 images of nutrition information
crawler.crawl(keyword="nutrition label food packaging", max_num=50)

# Show one of the downloaded images
img = Image.open("nutrition_labels/000001.jpg")
img.show()