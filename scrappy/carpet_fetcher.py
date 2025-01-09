import requests
from bs4 import BeautifulSoup
import csv
import os

BASE_URL = "https://www.ecarpetgallery.com/eu_en"
CATEGORY_URL = BASE_URL + "/shop-all-rugs/?p={}"

OUTPUT_CSV = "carpets.csv"
IMAGES_FOLDER = os.path.join("..", "datasets", "carpets")
MAX_PAGES = 977


# NOTE: Do not remove, otherwise the website blocks any access, this way we are giving cookies
#       to mimic a user accessing the website
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/109.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
})
try:
    home_resp = SESSION.get("https://www.ecarpetgallery.com", timeout=10)
    home_resp.raise_for_status()
except Exception as e:
    print(f"Warm-up request failed: {e}")

if not os.path.exists(IMAGES_FOLDER):
    os.makedirs(IMAGES_FOLDER)

def download_image(image_url, filename):
    """
    Downloads an image from 'image_url' and saves it as 'filename'.
    """
    try:
        resp = SESSION.get(image_url, stream=True, timeout=20)
        resp.raise_for_status()
        with open(filename, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
    except Exception as e:
        print(f"Failed to download {image_url}: {e}")


def scrape_one_page(page_number, debug=False):
    """
    Scrapes the given page number and returns a list of dictionaries:
      [
        {
          'carpet_name': ...,
          'carpet_material': ...,
          'carpet_weave': ...,
          'price': ...,
          'image_url': ...,
        },
        ...
      ]
    """
    url = CATEGORY_URL.format(page_number)
    print(f"Scraping page {page_number}/{MAX_PAGES} -> {url}")
    try:
        response = SESSION.get(url, timeout=15)
        if response.status_code != 200:
            print(f"Page {page_number} returned status code {response.status_code}")
            return []

        soup = BeautifulSoup(response.text, "html.parser")

        # image div and information divs are located in different blocks,
        # i tried to search with the next common block but it did not parse, so i went with the easiest solution
        product_items = zip(soup.find_all("div", {"class": "product details product-item-details"}), soup.find_all("div", {"class": "product-top"}))
        if not product_items:
            print(f"No products found on page {page_number}. Stopping.")
            return []

        page_data = []
        for info, image in product_items:
            try:
                # 1) Carpet name
                name_tag = info.select_one(
                    "h5.product.product-item-collection.product-item-name a.product-item-link"
                )
                carpet_name = name_tag.get_text(strip=True) if name_tag else None
                if debug:
                    print(f"    carpet_name = {carpet_name}")
                # 2) Carpet material
                material_tag = info.select_one(
                    "h5.product.product-item-origin-material.product-item-name a.product-item-link"
                )
                carpet_material = material_tag.get_text(strip=True) if material_tag else None
                if debug:
                    print(f"    material = {carpet_material}")
                # 3) Carpet weavery
                weave_tag = info.select_one(
                    "h5.product.product-item-weave.product-item-name a.product-item-link"
                )
                carpet_weave = weave_tag.get_text(strip=True) if weave_tag else None
                if debug:
                    print(f"    weave = {carpet_weave}")
                # 4) Size
                size_tag = info.select_one(
                    "h5.product.product-item-size.product-item-name a.product-item-link"
                )
                carpet_size = size_tag.get_text(strip=True) if size_tag else None
                if debug:
                    print(f"    {carpet_size}")
                # 4) Price
                price_tag = info.select_one(
                    "div.price-box.price-final_price span.map-actual-price.final-price "
                    "span.price-container.price-msrp_price.tax.weee span.price-wrapper"
                )
                carpet_price = price_tag["data-price-amount"] if price_tag and price_tag.has_attr("data-price-amount") else None
                if debug:
                    print(f"    price = {carpet_price}")
                # 5) Image URL
                img_tag = image.select_one(
                    "div.product-top a.product.photo.product-item-photo.has-hover-image img.img-responsive.product-image-photo.img-thumbnail"
                )
                image_url = img_tag["data-src"] if img_tag and img_tag.has_attr("data-src") else None
                if image_url:
                    # Ensure absolute URL if it's a relative path
                    if image_url.startswith("/"):
                        image_url = BASE_URL + image_url
                if debug:
                    print(f"    url = {image_url}")

                page_data.append({
                    "carpet_name": carpet_name,
                    "carpet_material": carpet_material,
                    "carpet_weave": carpet_weave,
                    "carpet_size": carpet_size,
                    "price": carpet_price,
                    "image_url": image_url
                })
            except Exception as e:
                print(f"Failed parsing a product on page {page_number}: {e}")

        return page_data
    except Exception as e:
        print(f"Failed to scrape page {page_number}: {e}")
        return []

def wrapper_download(i, record):
    image_id = f"{i:05d}.jpg"
    local_filename = os.path.join(IMAGES_FOLDER, image_id)

    if record["image_url"]:
        download_image(record["image_url"], local_filename)
    return i, image_id

def main():
    all_data = []
    for page in range(1, MAX_PAGES + 1):
        page_results = scrape_one_page(page)
        if not page_results:
            print(f"Error has occured while processing page {page}!")

        all_data.extend(page_results)
    
    print(f"Total carpets found: {len(all_data)}")

    for i, record in enumerate(all_data):
        image_id = f"{i:05d}.jpg"
        local_filename = os.path.join(IMAGES_FOLDER, image_id)
        record["id"] = i
        record["image_local_path"] = "None"

        if record["image_url"]:
            download_image(record["image_url"], local_filename)
            record["image_local_path"] = image_id

    with open(os.path.join("..", "datasets", OUTPUT_CSV), "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "name", "material", "weave", "size", "price", "image_url", "image_filename"])
        for record in all_data:
            writer.writerow([
                record["id"],
                record["carpet_name"],
                record["carpet_material"],
                record["carpet_weave"],
                record["carpet_size"],
                record["price"],
                record["image_url"] if record["image_url"] else "",
                record["image_local_path"] 
            ])

    print("Scraping complete!")
    print(f"Data saved to {OUTPUT_CSV}, images saved to folder '{IMAGES_FOLDER}'.")

if __name__ == "__main__":
    main()
